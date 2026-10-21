/**
 * The /traces page of the web UI (docs/trace-format.md 9.1, 9.3).
 *
 * Chosen in the CaP-X web UI route selection, so opening /traces never starts a
 * trial. Layout: top bar; a resizable left pane (task text and the causal
 * event timeline); a right pane split between media and the five detail
 * tabs. The URL holds `run`, `episode` and `event` (and `legacy_run`,
 * `legacy_trial` for runs without a trace) so a view can be shared.
 *
 * Live mode keeps the newest event selected and polls about every second
 * while the episode is running; any manual selection switches to Browse, which
 * stops following new events but still re-reads the episode status every 5 s.
 * When the episode stops running, events and status are polled for a short
 * grace period so the tail and late events are read (live.ts). Nothing here
 * can pause or stop an experiment. A poll tick never aborts the read still in
 * flight unless it has been in flight for 60 s (hooks.ts useAsync `poll`).
 *
 * Runner records (the top bar's verdict) are read with the run and read again
 * on 更新, when another episode of the same run is opened, at the start and
 * end of the grace period, and every 30 s while this episode has no record yet
 * and the run was seen with a running episode (this one included) less than a
 * minute before: a trial's record is appended only when its last attempt ends.
 * The top bar says when the records are still being read
 * or could not be read.
 *
 * Events whose data the writer moved to an asset are completed on demand
 * (dataAsset.ts): the events a pane needs for the selection are loaded and
 * merged before the tree is built.
 *
 * A run's unscoped log (events with no episode, such as the runner's
 * preflight requests) opens like an episode under the id unscoped.<pid>; its
 * summary has status "unscoped" and the page shows no task text for it.
 *
 * Related: hooks.ts (paging, polling, URL), model.ts (tree), components/*.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { traceApi } from './api';
import { applyDataAssets, dataAssetTargets, withTargetsLoading } from './dataAsset';
import { nextFocusToken } from './focus';
import { fmtClock } from './format';
import { useAsync, useDataAssets, useEpisodeEvents, useInterval, useStoredState, useUrlParams } from './hooks';
import { LABELS } from './labels';
import { GRACE_MAX_MS, RUNNER_POLL_MS, graceActive, live3dState, pollPlan, runnerPollActive, summaryDigest, type GraceStart, type Live3dState } from './live';
import { buildEpisodeModel, liveTarget, resolveSelection, selectionEventId, type TNode } from './model';
import { episodeKeyOf, runnerRecordFor } from './runner';
import { trialVideos } from './playback';
import { dataOf } from './types';
import { DetailTabs, type RangeFocus } from './components/DetailTabs';
import { ErrorBoundary } from './components/ErrorBoundary';
import { LegacyTrialView, type LegacyTrialRef } from './components/LegacyView';
import { MediaPane } from './components/MediaPane';
import { Picker } from './components/Picker';
import { SimplePicker, SimpleViewer } from './components/SimpleViewer';
import { Splitter } from './components/Splitter';
import { Timeline } from './components/Timeline';
import { TopBar, type PollStatus } from './components/TopBar';
import { Badge, ErrorText, SmallButton, Spinner } from './components/common';

function parseLegacyTrial(v: string | null, root: string | null): LegacyTrialRef | null {
  if (!v) return null;
  const parts = v.split('/');
  if (parts.length !== 3 || parts.some((p) => !p)) return null;
  const r = root !== null && /^\d+$/.test(root) ? Number(root) : null;
  return { root: r, task: parts[0], condition: parts[1], trialDir: parts[2] };
}

export default function TraceViewerPage() {
  const [params, setParams] = useUrlParams();
  const runId = params.get('run');
  const episodeId = params.get('episode');
  const eventParam = params.get('event');
  const legacyRun = params.get('legacy_run');
  const detailed = params.get('view') === 'details';
  const legacyTrial = parseLegacyTrial(params.get('legacy_trial'), params.get('legacy_root'));

  useEffect(() => {
    document.title = 'CaP-X Trace Viewer';
  }, []);

  const config = useAsync((s) => traceApi.config(s), []);
  const summary = useAsync(runId && episodeId ? (s) => traceApi.episode(runId, episodeId, s) : null, [runId, episodeId]);
  const runDetail = useAsync(runId ? (s) => traceApi.run(runId, s) : null, [runId]);
  const status = summary.data?.status ?? null;
  const running = status === 'running';
  const epKey = runId && episodeId ? `${runId}\u0000${episodeId}` : '';

  // Live: newest event selected; start in Browse when the URL names an event.
  const [live, setLive] = useState<boolean>(() => !new URLSearchParams(window.location.search).get('event'));
  // Polling after the episode stopped running (tail and late events); see live.ts.
  const [grace, setGrace] = useState<GraceStart | null>(null);
  const plan = pollPlan({ live, running, grace: grace !== null });
  const events = useEpisodeEvents(runId, episodeId, plan.events && !!runId && !!episodeId);
  useInterval(summary.poll, plan.summaryMs ?? 1000, plan.summaryMs !== null && !!runId && !!episodeId);

  // Summary reads, and how many in a row were identical (ends the grace period).
  const [summaryReads, setSummaryReads] = useState<{ key: string; n: number; stable: number; last: string | null }>({ key: '', n: 0, stable: 0, last: null });
  useEffect(() => {
    if (!summary.data) return;
    const d = summaryDigest(summary.data);
    setSummaryReads((p) => (p.key !== epKey ? { key: epKey, n: 1, stable: 0, last: d } : { key: epKey, n: p.n + 1, stable: p.last === d ? p.stable + 1 : 0, last: d }));
    // epKey only labels the read; a new episode brings new summary data anyway.
  }, [summary.data]);

  // When the run was last seen with a running episode (this one, or one in the run's list):
  // while this episode has no runner record, the records are re-read for a while after that.
  const [runSeenRunningAt, setRunSeenRunningAt] = useState<{ runId: string; at: number } | null>(null);

  // When the episode stops running: fetch the tail at once (the pager reads again after a
  // fetch in flight), refresh the runner records, and keep polling for the grace period.
  const prevStatus = useRef<{ key: string; status: string | null }>({ key: '', status: null });
  const refreshEvents = events.refresh;
  const reloadRun = runDetail.reload;
  useEffect(() => {
    const prev = prevStatus.current;
    if (prev.key === epKey && prev.status === 'running' && status !== null && status !== 'running') {
      setGrace({ startedAt: Date.now(), passesAtStart: events.passes, summaryReadsAtStart: summaryReads.n });
      if (runId) setRunSeenRunningAt({ runId, at: Date.now() });
      refreshEvents();
      reloadRun();
    }
    if (prev.key !== epKey) setGrace(null);
    prevStatus.current = { key: epKey, status };
    // Only a status change starts the grace period; the counters are read as of that render.
    // `runId` is part of `epKey`.
  }, [status, epKey, refreshEvents, reloadRun]);
  // Another episode of the same run: its runner record may be newer than the last read
  // (a new run is read by useAsync itself).
  const openedEpisode = useRef<{ runId: string | null; epKey: string }>({ runId, epKey });
  useEffect(() => {
    const prev = openedEpisode.current;
    openedEpisode.current = { runId, epKey };
    if (runId && prev.runId === runId && prev.epKey !== epKey) reloadRun();
  }, [runId, epKey, reloadRun]);
  useEffect(() => {
    if (!grace) return;
    const check = () => {
      const active = graceActive(grace, {
        now: Date.now(),
        passes: events.passes,
        quietPasses: events.quietPasses,
        summaryReads: summaryReads.n,
        stableSummaryReads: summaryReads.stable,
      });
      if (!active) {
        setGrace(null);
        reloadRun();
      }
    };
    check();
    const id = window.setInterval(check, 1000);
    return () => window.clearInterval(id);
  }, [grace, events.passes, events.quietPasses, summaryReads.n, summaryReads.stable, reloadRun]);

  // Events with the data the writer moved to assets merged back (loaded on demand below).
  // The open episode. Its files are fetched under the generation of events.jsonl, so after
  // the file was replaced nothing cached for the old trace is shown (api.ts episodeFileUrl).
  const ep = useMemo(() => (runId && episodeId ? { runId, episodeId, generation: events.generation } : null), [runId, episodeId, events.generation]);
  const assets = useDataAssets(ep, String(events.resets));
  const merged = useMemo(() => applyDataAssets(events.events, assets.states), [events.events, assets.states]);
  const model = useMemo(() => buildEpisodeModel(merged), [merged]);
  const selection = useMemo(() => resolveSelection(model, eventParam), [model, eventParam]);
  const assetTargets = useMemo(() => dataAssetTargets(model, selection), [model, selection]);
  const ensureAssets = assets.ensure;
  useEffect(() => ensureAssets(assetTargets), [assetTargets, ensureAssets]);
  // What the panes are told: the targets requested by the effect above count as loading from
  // this render on (an oversize event with no state is one the page does not load itself).
  const shownAssetStates = useMemo(() => withTargetsLoading(assets.states, assetTargets), [assets.states, assetTargets]);

  // The runner record of this episode: while it is missing and the run may still append it
  // (the run was seen with a running episode less than a minute ago), re-read slowly.
  const startWall = (model.episodeStart?.wall_time ?? null) ?? summary.data?.start_wall_time ?? null;
  const runnerRecords = runDetail.data?.runner_records ?? null;
  const hasRunnerRecord = useMemo(
    () =>
      runnerRecordFor(runnerRecords ?? [], episodeKeyOf(model.episodeStart ? dataOf(model.episodeStart, 'episode_start') : null, summary.data, startWall)).record !==
      null,
    [runnerRecords, model.episodeStart, summary.data, startWall],
  );
  useEffect(() => {
    if (runId && runDetail.data?.episodes?.some((e) => e && e.status === 'running')) setRunSeenRunningAt({ runId, at: Date.now() });
  }, [runDetail.data, runId]);
  useInterval(
    runDetail.poll,
    RUNNER_POLL_MS,
    runnerPollActive({
      open: !!runId && !!episodeId && status !== null && status !== 'unscoped',
      thisRunning: running || grace !== null,
      hasRecord: hasRunnerRecord,
      runSeenRunningAt: runSeenRunningAt && runSeenRunningAt.runId === runId ? runSeenRunningAt.at : null,
      now: Date.now(),
    }),
  );

  const [scrollToken, setScrollToken] = useState(0);
  const [tab, setTab] = useState(0);
  const [focus, setFocus] = useState<RangeFocus | null>(null);
  const [pickerOpen, setPickerOpen] = useState(false);
  const [playbackEventId, setPlaybackEventId] = useState<string | null>(null);
  const videos = useMemo(() => runId ? trialVideos(runId,
    runnerRecordFor(runnerRecords ?? [], episodeKeyOf(model.episodeStart ? dataOf(model.episodeStart, 'episode_start') : null, summary.data, startWall)),
    config.data?.runner_roots ?? []) : [], [runId, runnerRecords, model.episodeStart, summary.data, startWall, config.data]);

  // Live mode: follow the newest event.
  useEffect(() => {
    if (!live || !runId || !episodeId) return;
    const t = detailed ? liveTarget(model) : !eventParam && model.blocks.length ? {eventId: selectionEventId(model.blocks[0])} : null;
    if (t && t.eventId !== eventParam) {
      setParams({ event: t.eventId });
      setScrollToken((x) => x + 1);
    }
  }, [live, detailed, model, eventParam, runId, episodeId, setParams]);

  // When the URL names an event that is loaded, bring it into view once.
  const scrolledFor = useRef<string | null>(null);
  useEffect(() => {
    if (!selection || scrolledFor.current === selection.eventId) return;
    scrolledFor.current = selection.eventId;
    setScrollToken((x) => x + 1);
  }, [selection]);

  const onSelect = useCallback(
    (node: TNode, eventId?: string) => {
      setPlaybackEventId(null);
      setLive(false);
      setParams({ event: eventId ?? selectionEventId(node) });
      setScrollToken((x) => x + 1);
    },
    [setParams],
  );

  const onFocusRange = useCallback(
    (eventId: string, path: string, start: number, end: number) => {
      setLive(false);
      setParams({ event: eventId });
      setTab(0);
      setFocus({ eventId, path, start, end, token: nextFocusToken() });
      setScrollToken((x) => x + 1);
    },
    [setParams],
  );

  const onPlaybackSelect = useCallback((node: TNode) => {
    onSelect(node);
    setPlaybackEventId(selectionEventId(node));
    setTab(node.kind === 'model_call' ? 0 : 2);
    setFocus(null);
  }, [onSelect]);

  // A range focus applies only to the call it was made for.
  useEffect(() => {
    if (focus && selection && !selection.node.events.some((e) => e.event_id === focus.eventId)) setFocus(null);
  }, [selection, focus]);

  const onLive = (v: boolean) => {
    setLive(v);
    if (v) setScrollToken((x) => x + 1);
  };

  // Live 3D: only when this episode runs (status re-read in Browse too) and its recorded Viser
  // port is the one the proxy serves.
  const viserPort =
    (model.episodeStart ? dataOf(model.episodeStart, 'episode_start').env?.viser_port : null) ?? summary.data?.viser_port ?? null;
  const viewerOnly = config.data?.viewer_only === true;
  const viserWanted = running && viserPort !== null && viserPort !== undefined && !viewerOnly;
  const viser = useAsync(viserWanted ? (s) => traceApi.viser(s) : null, [running, viserPort, viewerOnly]);
  // Re-checked every 5 s while the episode runs, so the tab goes away when Viser stops.
  useInterval(viser.poll, 5000, viserWanted);
  const live3d: Live3dState = useMemo(
    () => live3dState({ running, viserPort, viewerOnly, viser: viser.data, viserError: viser.error }),
    [running, viserPort, viewerOnly, viser.data, viser.error],
  );

  const graceOn = grace !== null;
  const pollStatus: PollStatus = useMemo(() => {
    if (events.error) return { text: `取得エラー: ${events.error}`, tone: 'red' };
    if (events.replaced && events.loading) return { text: 'events.jsonl が置き換えられたため読み直し中', tone: 'amber' };
    if (events.loading) return { text: '読み込み中', tone: 'gold' };
    if (graceOn) return { text: `終了後に届くイベントを確認中（最長 ${GRACE_MAX_MS / 1000} 秒）`, tone: 'gold' };
    if (running && live) return { text: `ライブ更新中（約1秒ごと）${events.lastFetch ? ` · ${fmtClock(events.lastFetch)}` : ''}`, tone: 'gold' };
    if (running && !live) return { text: '更新停止中（Browse。状態は 5 秒ごとに確認）', tone: 'neutral' };
    if (status === 'incomplete') return { text: '更新停止（記録が止まっています）', tone: 'neutral' };
    if (status === 'unscoped') return { text: '自動更新なし（「更新」で再取得）', tone: 'neutral' };
    if (status) return { text: '更新停止（エピソード終了）', tone: 'neutral' };
    return { text: '更新停止', tone: 'neutral' };
  }, [events.error, events.loading, events.replaced, events.lastFetch, graceOn, running, live, status]);

  // Resizable panes (sizes are a per-viewer convenience kept in localStorage).
  const [leftWidth, setLeftWidth] = useStoredState<number>('capx-trace-left-width', 460);
  const [mediaPct, setMediaPct] = useStoredState<number>('capx-trace-media-pct', 42);
  const mainRef = useRef<HTMLDivElement>(null);
  const rightRef = useRef<HTMLDivElement>(null);
  const clampLeft = (w: number) => {
    const total = mainRef.current?.getBoundingClientRect().width ?? 1400;
    return Math.round(Math.max(280, Math.min(total - 420, w)));
  };
  const clampPct = (p: number) => Math.max(12, Math.min(85, p));

  const pick = (r: string, e: string) => {
    setPickerOpen(false);
    setLive(true);
    setFocus(null);
    setParams({ run: r, episode: e, event: null, legacy_run: null, legacy_trial: null, legacy_root: null }, true);
  };
  const pickLegacy = (r: string, t: LegacyTrialRef) => {
    setPickerOpen(false);
    setParams(
      { run: null, episode: null, event: null, legacy_run: r, legacy_trial: `${t.task}/${t.condition}/${t.trialDir}`, legacy_root: t.root !== null ? String(t.root) : null },
      true,
    );
  };

  const picker = !detailed ? <SimplePicker onPick={pick} onDetails={() => setParams({view: 'details'})} /> : (
    <Picker
      config={config.data}
      configError={config.error}
      currentRun={runId}
      currentEpisode={episodeId}
      onPick={pick}
      onPickLegacy={pickLegacy}
      onClose={runId && episodeId ? () => setPickerOpen(false) : undefined}
    />
  );

  const showLegacy = !ep && legacyRun && legacyTrial;
  const simple = !detailed && !showLegacy;
  const simpleMatch = runnerRecordFor(runnerRecords ?? [], episodeKeyOf(model.episodeStart ? dataOf(model.episodeStart, 'episode_start') : null, summary.data, startWall));
  const simpleOutcome = running ? '実行中' : simpleMatch.isFinal && !simpleMatch.unverified && simpleMatch.record?.outcome?.task_success === true ? '成功' : simpleMatch.isFinal && !simpleMatch.unverified && simpleMatch.record?.outcome?.task_success === false ? '未達成' : '結果未確認';

  // Font ligatures are off: the viewer shows recorded text as it was sent ("->" must not
  // turn into an arrow, "!=" into "≠").
  return (
    <div className="h-full flex flex-col bg-surface text-text-primary [font-variant-ligatures:none]">
      <header className="flex-shrink-0 flex flex-wrap items-center gap-3 px-5 py-3 border-b border-surface-border bg-surface-raised">
        <span className="font-semibold">{simple ? '実験を見る' : '実験の詳細'}</span>
        {simple && ep && <><span className="text-sm text-text-secondary">{summary.data?.task ?? ''} · {summary.data?.condition ?? ''}</span><span className="text-sm text-accent">試行全体の最終結果：{simpleOutcome}</span></>}
        {simple && summary.data?.trace_complete === false && <span className="text-sm text-amber-200">記録に欠落があります</span>}
        <div className="ml-auto flex gap-2">
          <SmallButton onClick={() => setPickerOpen(true)}>試行を選ぶ</SmallButton>
          <SmallButton onClick={() => setParams({view: detailed ? null : 'details'})}>{detailed ? 'シンプル表示に戻る' : '詳細を見る'}</SmallButton>
        </div>
      </header>
      {!simple && <ErrorBoundary
        label="上部バー"
        resetKey={`${runId ?? ''}|${episodeId ?? ''}`}
        fallbackExtra={<SmallButton onClick={() => setPickerOpen(true)}>run / episode を選ぶ</SmallButton>}
      >
        <TopBar
          runLabel={showLegacy ? legacyRun : runId}
          episodeLabel={showLegacy ? `${legacyTrial.task}/${legacyTrial.condition}/${legacyTrial.trialDir}` : episodeId}
          episodeId={ep ? episodeId : null}
          summary={summary.data}
          model={ep ? model : null}
          runnerRecords={runnerRecords ?? []}
          runnerRead={{ loaded: runDetail.data !== null, loading: runDetail.loading, error: runDetail.error }}
          runnerRecordsInfo={runDetail.data?.runner_records_info ?? null}
          startWall={startWall}
          live={live}
          onLive={onLive}
          poll={pollStatus}
          lastFetch={events.lastFetch}
          reader={{
            corruptLines: events.corruptLines,
            tornTail: events.tornTail,
            eof: events.eof,
            count: events.events.length,
            replaced: events.replaced,
            error: events.error,
          }}
          onOpenPicker={() => setPickerOpen(true)}
          onRefresh={() => {
            events.refresh();
            summary.reload();
            runDetail.reload();
          }}
          legacy={showLegacy ? <Badge tone="amber">{LABELS.legacySection}</Badge> : undefined}
          assetStates={shownAssetStates}
          onRetryAsset={assets.retry}
        />
      </ErrorBoundary>}
      {pickerOpen && (
        <div className="fixed inset-0 z-40 bg-black/60 flex items-start justify-center pt-14" onClick={() => setPickerOpen(false)}>
          <div className="w-[min(1200px,94vw)] h-[80vh] rounded-lg border border-surface-border-light shadow-2xl overflow-hidden animate-scale-in" onClick={(e) => e.stopPropagation()}>
            {picker}
          </div>
        </div>
      )}
      {showLegacy ? (
        <LegacyTrialView runId={legacyRun} trial={legacyTrial} />
      ) : !ep ? (
        <div className="flex-1 min-h-0">{picker}</div>
      ) : simple ? (
        <ErrorBoundary label="動画と入力" resetKey={epKey}>
          <SimpleViewer model={model} ep={ep} selection={selection} videos={videos} onSelect={onPlaybackSelect} error={events.error ?? summary.error ?? config.error} />
        </ErrorBoundary>
      ) : (
        <main ref={mainRef} className="flex-1 min-h-0 flex">
          <div className="flex-shrink-0 min-w-0 border-r border-surface-border bg-surface" style={{ width: leftWidth }}>
            {summary.error && !summary.data ? (
              <div className="p-4">
                <ErrorText>エピソードを開けません: {summary.error}</ErrorText>
              </div>
            ) : events.loading && events.events.length === 0 ? (
              <div className="p-4">
                <Spinner label="イベントを読み込み中" />
              </div>
            ) : events.error && events.events.length === 0 ? (
              <div className="p-4">
                <ErrorText>イベントを読み込めません: {events.error}</ErrorText>
              </div>
            ) : (
              <Timeline
                model={model}
                ep={ep}
                running={running}
                selectedKey={selection?.node.key ?? null}
                selectedEventId={selection?.eventId ?? null}
                onSelect={onSelect}
                scrollToken={scrollToken}
                showTaskText={status !== 'unscoped'}
              />
            )}
          </div>
          <Splitter
            orientation="vertical"
            ariaLabel="左ペインの幅"
            onDrag={(x) => {
              const left = mainRef.current?.getBoundingClientRect().left ?? 0;
              setLeftWidth(clampLeft(x - left));
            }}
            onKeyStep={(d) => setLeftWidth(clampLeft(leftWidth + d))}
          />
          <div ref={rightRef} className="flex-1 min-w-0 flex flex-col">
            <div className="min-h-0 overflow-hidden" style={{ height: `${mediaPct}%` }}>
              <ErrorBoundary label="メディア" resetKey={selection?.eventId ?? null}>
                <MediaPane model={model} ep={ep} selection={selection} live3d={live3d} follow={live && running} videos={videos} onPlaybackSelect={onPlaybackSelect} />
              </ErrorBoundary>
            </div>
            <Splitter
              orientation="horizontal"
              ariaLabel="メディアと詳細の境界"
              onDrag={(y) => {
                const r = rightRef.current?.getBoundingClientRect();
                if (!r || r.height <= 0) return;
                setMediaPct(clampPct(((y - r.top) / r.height) * 100));
              }}
              onKeyStep={(d) => {
                const h = rightRef.current?.getBoundingClientRect().height ?? 800;
                setMediaPct(clampPct(mediaPct + (d / h) * 100));
              }}
            />
            <div className="flex-1 min-h-0">
              <DetailTabs
                revealPrompt={playbackEventId !== null && playbackEventId === selection?.eventId}
                model={model}
                ep={ep}
                summary={summary.data}
                selection={selection}
                missingEventId={eventParam && !selection && events.eof && !events.loading ? eventParam : null}
                running={running}
                onSelect={onSelect}
                onFocusRange={onFocusRange}
                focus={focus}
                tab={tab}
                onTab={setTab}
                assetStates={shownAssetStates}
                onRetryAsset={assets.retry}
                assetEvents={assetTargets}
              />
            </div>
          </div>
        </main>
      )}
    </div>
  );
}
