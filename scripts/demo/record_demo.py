"""Record the README demo videos of the viewer by driving a running server in headless Chromium.

Two scripted sessions are recorded with Playwright's video recording, and the README
screenshots are taken in a third session:

- simple: scenario_simple.js, on the start page (choose the experiment, open a trial, play
  its video, switch to the model output and the execution log, jump to another code
  execution, return to the prompt).
- details: scenario_details.js, on the simple view of one trial (open the detailed view,
  select an API call and a service call, pick a model request in the video panel, search
  the timeline).
- screens: 01_select_run.png to 06_details_view.png (start page, trial list, one code
  execution with its prompt, model output, execution log, detailed view).

demo_helpers.js adds a red dot that moves to each element before it is clicked. The
scenarios find elements by their (Japanese) interface text and by the labels passed from
this script. The defaults match the trial cube_stack__M3__s1__i1__a1 of the run
viewer_demo, two Cube Stack M3 trials recorded for this README (CaP-X distributed config,
model qwen3.8-flash-next-iq3_s served locally). For another trace, pass labels that exist
in it.

Each video recording is cut to the scripted part and written as <name>.mp4 (H.264, 30 fps) and
<name>.gif (README preview) into --out-dir. Needs ffmpeg on PATH and the demo extra
(`uv pip install -e '.[demo]'`, then `playwright install chromium`).

Usage:
    .venv/bin/python scripts/demo/record_demo.py --base-url http://127.0.0.1:8290 \
        --out-dir docs/media
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import time
from pathlib import Path
from urllib.parse import urlencode

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
VIEWPORT = {"width": 1440, "height": 900}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base-url", default="http://127.0.0.1:8290", help="running viewer server")
    ap.add_argument("--run", default="viewer_demo")
    ap.add_argument("--episode", default="cube_stack__M3__s1__i1__a1",
                    help="trial opened by the details recording (the simple recording and the "
                         "screenshots open the trial chosen by --trial-label)")
    ap.add_argument("--trial-label", default="seed 1 ·",
                    help="text of the trial button clicked in the trial list")
    ap.add_argument("--block-label", default="3回目", help="code execution chosen in the simple view")
    ap.add_argument("--api-call", default="get_object_pose")
    ap.add_argument("--service-call", default="sam3.sam3_seg_fn")
    ap.add_argument("--model-event", default="0.58s vdm · image_diff",
                    help="model request clicked in the video panel")
    ap.add_argument("--search", default="goto_pose")
    ap.add_argument("--out-dir", type=Path, default=Path("docs/media"))
    ap.add_argument("--prefix", default="viewer_demo_")
    ap.add_argument("--only", choices=["simple", "details", "screens"], action="append",
                    help="record only this demo or only the screenshots (repeatable)")
    return ap.parse_args()


def record(browser, url: str, scenario: Path, params: dict, workdir: Path) -> tuple[Path, float, float]:
    """Run one scenario and return the raw webm with the start and end of the scripted part."""
    ctx = browser.new_context(viewport=VIEWPORT, locale="ja-JP", record_video_dir=str(workdir),
                              record_video_size=VIEWPORT)
    page = ctx.new_page()
    t0 = time.time()
    page.goto(url, wait_until="networkidle")
    page.wait_for_timeout(800)
    page.evaluate((HERE / "demo_helpers.js").read_text())
    start = time.time() - t0
    page.evaluate(f"({scenario.read_text()})({json.dumps(params)})")
    page.wait_for_timeout(300)
    end = time.time() - t0
    video = page.video
    ctx.close()
    return Path(video.path()), start, end


def screenshots(browser, base: str, args: argparse.Namespace) -> None:
    """Take the six README screenshots of one trial."""
    page = browser.new_page(viewport=VIEWPORT, locale="ja-JP")
    page.goto(base, wait_until="networkidle")
    page.wait_for_timeout(500)
    shot = lambda name: page.screenshot(path=str(args.out_dir / name))
    shot("01_select_run.png")
    page.get_by_label("実験を選ぶ").select_option(label=next(
        o for o in page.get_by_label("実験を選ぶ").locator("option").all_inner_texts() if args.run in o))
    page.wait_for_timeout(800)
    shot("02_trial_list.png")
    page.get_by_role("button", name=args.trial_label).first.click()
    page.wait_for_timeout(1500)
    shot("03_trial_detail.png")
    for tab, name in (("モデル出力", "04_model_output.png"), ("実行ログ", "05_execution_log.png")):
        page.get_by_role("button", name=tab, exact=True).click()
        page.wait_for_timeout(700)
        shot(name)
    page.get_by_role("button", name="詳細を見る").click()
    page.wait_for_timeout(2000)
    # The shortest matching element is the timeline card of the call, not a code block
    # whose source mentions the same name (same rule as demo_helpers.js).
    page.evaluate("""(t) => {
      const els = [...document.querySelectorAll('button,[role="button"]')]
        .filter((e) => e.textContent.includes(t));
      const el = els.reduce((a, e) => (!a || e.textContent.length < a.textContent.length ? e : a), null);
      el.scrollIntoView({ block: 'nearest' });
      el.click();
    }""", args.api_call)
    page.wait_for_timeout(1200)
    shot("06_details_view.png")
    page.close()


def encode(raw: Path, start: float, end: float, out_mp4: Path, out_gif: Path) -> None:
    cut = ["-ss", f"{max(start - 0.1, 0):.2f}", "-to", f"{end:.2f}", "-i", str(raw)]
    subprocess.run(["ffmpeg", "-v", "error", "-y", *cut, "-c:v", "libx264", "-crf", "20",
                    "-preset", "slow", "-pix_fmt", "yuv420p", "-r", "30", "-movflags", "+faststart",
                    "-an", str(out_mp4)], check=True)
    gif_filter = ("fps=10,scale=960:-1:flags=lanczos,split[a][b];"
                  "[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer:bayer_scale=4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", *cut, "-vf", gif_filter, str(out_gif)], check=True)


def main() -> None:
    args = parse_args()
    base = args.base_url.rstrip("/") + "/traces"
    jobs = {
        "simple": (base, HERE / "scenario_simple.js"),
        "details": (base + "?" + urlencode({"run": args.run, "episode": args.episode}),
                    HERE / "scenario_details.js"),
    }
    params = {"run": args.run, "trialLabel": args.trial_label, "blockLabel": args.block_label,
              "apiCall": args.api_call, "serviceCall": args.service_call,
              "modelEvent": args.model_event, "search": args.search}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp, sync_playwright() as p:
        browser = p.chromium.launch()
        for name in args.only or [*jobs, "screens"]:
            if name == "screens":
                screenshots(browser, base, args)
                print(f"screens: {args.out_dir}/01_select_run.png ... 06_details_view.png")
                continue
            url, scenario = jobs[name]
            raw, start, end = record(browser, url, scenario, params, Path(tmp) / name)
            out = args.out_dir / f"{args.prefix}{name}"
            encode(raw, start, end, out.with_suffix(".mp4"), out.with_suffix(".gif"))
            print(f"{name}: {out}.mp4 and .gif ({end - start + 0.1:.1f} s)")
        browser.close()


if __name__ == "__main__":
    main()
