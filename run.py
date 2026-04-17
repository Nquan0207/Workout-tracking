import argparse

from .pipeline.analyze import Analyzer

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--video', help='input video path')
    src.add_argument('--camera', action='store_true', help='use live camera feed')
    ap.add_argument('--exercise', default='squat', choices=['squat','pushup','bicep_curl','pullup'])
    ap.add_argument('--overlay', action='store_true', help='write overlay mp4')
    ap.add_argument('--overlay-out', default='overlay.mp4')
    ap.add_argument('--json-out', default='result.json')
    ap.add_argument('--camera-index', type=int, default=0, help='camera device index (default: 0)')
    ap.add_argument('--live-recording', help='when using --camera, optionally save the overlaid video to this path')
    ap.add_argument('--voice', action='store_true', help='announce each counted rep out loud')
    ap.add_argument('--voice-name', help='system voice name to use for rep announcements')
    args = ap.parse_args()

    an = Analyzer(exercise=args.exercise, overlay=args.overlay, voice=args.voice, voice_name=args.voice_name)
    if args.camera:
        if not args.overlay:
            print('Tip: pass --overlay to see the live visualization window. Press q or ESC to quit.')
        res = an.run_live(src=args.camera_index,
                          json_out=(args.json_out or None),
                          out_overlay_path=(args.live_recording if args.overlay and args.live_recording else None))
    else:
        res = an.run_on_video(args.video, out_overlay_path=(args.overlay_out if args.overlay else None), json_out=args.json_out)
    print('Done. Total reps:', res['summary']['total_reps'])
