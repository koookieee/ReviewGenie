"""
Trajectory Viewer Server
Serves the viewer app and dynamically lists available trajectories.
"""
import json
import os
from pathlib import Path
from aiohttp import web

VIEWER_DIR = Path(__file__).parent
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", VIEWER_DIR / "results"))


async def handle_index(request):
    return web.FileResponse(VIEWER_DIR / "index.html")


async def handle_trajectories(request):
    """Dynamically list all trajectory.json files in results/ and viewer dir.

    Handles both flat and nested (paper_id/attempt_N/trajectory.json) layouts.
    """
    entries = []
    if RESULTS_DIR.is_dir():
        for top_dir in sorted(RESULTS_DIR.iterdir(), reverse=True):
            if not top_dir.is_dir():
                continue
            # Flat layout: results/run_name/trajectory.json
            traj = top_dir / "trajectory.json"
            if traj.is_file():
                entries.append({
                    "label": top_dir.name,
                    "url": f"/results/{top_dir.name}/trajectory.json",
                })
                continue
            # Nested layout: results/paper_id/attempt_N/trajectory.json
            for attempt_dir in sorted(top_dir.iterdir(), reverse=True):
                if not attempt_dir.is_dir():
                    continue
                traj = attempt_dir / "trajectory.json"
                if traj.is_file():
                    label = f"{top_dir.name}/{attempt_dir.name}"
                    url = f"/results/{top_dir.name}/{attempt_dir.name}/trajectory.json"
                    entries.append({"label": label, "url": url})

    # Also pick up standalone trajectory_*.json files in viewer dir
    for f in sorted(VIEWER_DIR.glob("trajectory_*.json"), reverse=True):
        entries.append({
            "label": f.stem,
            "url": f"/{f.name}",
        })
    return web.Response(
        text=json.dumps(entries),
        content_type="application/json"
    )


async def handle_result_file(request):
    """Serve files from results directory (flat or one level nested)."""
    run = request.match_info["run"]
    filename = request.match_info["filename"]
    # Try direct path first
    path = RESULTS_DIR / run / filename
    if path.is_file():
        return web.FileResponse(path)
    raise web.HTTPNotFound()


async def handle_nested_result_file(request):
    """Serve files two levels deep: results/paper_id/attempt_N/filename."""
    paper = request.match_info["paper"]
    attempt = request.match_info["attempt"]
    filename = request.match_info["filename"]
    path = RESULTS_DIR / paper / attempt / filename
    if not path.is_file():
        raise web.HTTPNotFound()
    return web.FileResponse(path)


app = web.Application()
app.router.add_get("/", handle_index)
app.router.add_get("/trajectories.json", handle_trajectories)
app.router.add_get("/results/{run}/{filename}", handle_result_file)
app.router.add_get("/results/{paper}/{attempt}/{filename}", handle_nested_result_file)
app.router.add_static("/", VIEWER_DIR)

if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8282
    print(f"Trajectory viewer on :{port}")
    web.run_app(app, port=port, print=None)
