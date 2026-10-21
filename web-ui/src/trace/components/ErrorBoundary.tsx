/**
 * Keeps one part of the trace viewer from blanking the whole page when a
 * recorded event has a shape the viewer did not expect (React unmounts the
 * whole root on an uncaught render error). The part shows a short notice with
 * the error text instead; `resetKey` clears it when the selection changes.
 * Wrapped around the top bar, each timeline card, the media pane and each
 * detail tab.
 *
 * Related: TraceViewerPage.tsx, Timeline.tsx, DetailTabs.tsx.
 */

import { Component, type ReactNode } from 'react';

interface Props {
  /** What this part shows, for the notice (e.g. 「このカード」). */
  label: string;
  /** Changing it clears a caught error (a new event was selected). */
  resetKey?: string | number | null;
  /** Shown after the notice, for example a control the failed part carried. */
  fallbackExtra?: ReactNode;
  children: ReactNode;
}

interface State {
  error: string | null;
  key: string | number | null | undefined;
}

export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null, key: this.props.resetKey };

  static getDerivedStateFromError(err: unknown): Partial<State> {
    return { error: err instanceof Error ? `${err.name}: ${err.message}` : String(err) };
  }

  static getDerivedStateFromProps(props: Props, state: State): Partial<State> | null {
    return props.resetKey !== state.key ? { error: null, key: props.resetKey } : null;
  }

  render(): ReactNode {
    if (this.state.error === null) return this.props.children;
    return (
      <div role="alert" className="px-3 py-2 rounded-md border border-red-500/40 bg-red-500/5 text-xs text-red-300 break-all">
        {this.props.label}を表示できません（記録の形式がビューワーの想定と違います）: {this.state.error}
        {this.props.fallbackExtra ? <div className="mt-1">{this.props.fallbackExtra}</div> : null}
      </div>
    );
  }
}
