/**
 * ==============================================================================
 * 📐 [DEV 3] MATHEMATICAL OVERLAY ENGINE (app/components/SVGOverlay.tsx)
 * ==============================================================================
 * Purpose:
 *   This isolated component is entirely responsible for drawing the "Pause & Draw"
 *   illusion over top of the video player when Dev 1 pauses it.
 *
 * Responsibilities:
 *   1. Takes the JSON `visuals` object from Gemini.
 *   2. Maps relative (`0.0 - 1.0`) coordinates to Absolute Pixel coordinates.
 *   3. Uses `react-native-svg` to draw Lines, Paths, and Arcs (e.g., an angle
 *      showing "Bend elbow 30 degrees more").
 *
 * The Bounding Box Trap (CRITICAL WARNING):
 *   - The phone screen is likely `19.5:9`. The video is likely `16:9` or `4:3`.
 *   - Getting the Screen Width/Height does NOT map to the correct pixels if the 
 *     video is letterboxed (black bars at the top/bottom).
 *   - You MUST compute the height/width of the EXACT video rendering window, 
 *     and multiply Gemini's `0.5` center-point against the Video Width, NOT 
 *     the Screen Width. 
 *          ✅ correct_x = gemini_x * video_pixel_width
 *          ❌ wrong_x = gemini_x * Dimensions.get('window').width
 *
 * Development Workflow:
 *   - You DO NOT need to wait for Dev 2 to build the modal AI. Simply import
 *     the `app/data/mock_response.json` file to build your drawings immediately.
 *
 * How to test:
 *   1. In-app: On Recording screen tap "Test overlay (mock data)" to open
 *      Playback with mock visuals; the overlay is shown by default.
 *   2. Unit tests: Run `npx expo test` (or `npm test`) from project root;
 *      see app/components/SVGOverlay.test.tsx.
 * ==============================================================================
 */
import React from 'react';
import { View, StyleSheet, Dimensions } from 'react-native';
import Svg, { Line, Circle, Path, Text as SvgText } from 'react-native-svg';

export interface OverlayVector {
  start: [number, number];
  end: [number, number];
  color: string;
  label?: string;
}

export interface OverlayFocusPoint {
  x: number;
  y: number;
}

export type OverlayType = 'ANGLE_CORRECTION' | 'POSITION_MARKER' | 'PATH_TRACE';

export interface OverlayData {
  overlay_type?: OverlayType | string;
  focus_point?: OverlayFocusPoint;
  vectors?: OverlayVector[];
  path_points?: Array<[number, number]>;
}

export interface VideoLayout {
  width: number;
  height: number;
}

function clamp01(n: number) {
  return Math.max(0, Math.min(1, n));
}

function toPxX(x01: number, width: number) {
  return clamp01(x01) * width;
}

function toPxY(y01: number, height: number) {
  return clamp01(y01) * height;
}

function normalizeRadians(r: number) {
  const twoPi = Math.PI * 2;
  return ((r % twoPi) + twoPi) % twoPi;
}

function shortestDelta(a: number, b: number) {
  // signed delta from a -> b in [-pi, pi]
  const twoPi = Math.PI * 2;
  let d = normalizeRadians(b) - normalizeRadians(a);
  if (d > Math.PI) d -= twoPi;
  if (d < -Math.PI) d += twoPi;
  return d;
}

function arcPath(cx: number, cy: number, r: number, startRad: number, endRad: number) {
  const d = shortestDelta(startRad, endRad);
  const sweepFlag = d >= 0 ? 1 : 0;
  const largeArcFlag = Math.abs(d) > Math.PI ? 1 : 0;

  const x1 = cx + r * Math.cos(startRad);
  const y1 = cy + r * Math.sin(startRad);
  const x2 = cx + r * Math.cos(endRad);
  const y2 = cy + r * Math.sin(endRad);

  return `M ${x1} ${y1} A ${r} ${r} 0 ${largeArcFlag} ${sweepFlag} ${x2} ${y2}`;
}

export default function SVGOverlay({
  data,
  videoLayout,
}: {
  data?: OverlayData;
  videoLayout?: VideoLayout;
}) {

    // 1. Calculate the exact bounding box of the Dev 1's <Video> player.
    //    (Example math assumes a perfectly locked 16:9 video taking up full width)
    const screenWidth = Dimensions.get('window').width;
    const fallbackHeight = screenWidth * (16 / 9);

    const width = videoLayout?.width ?? screenWidth;
    const height = videoLayout?.height ?? fallbackHeight;

    const vectors = data?.vectors ?? [];
    const overlayType = data?.overlay_type ?? 'ANGLE_CORRECTION';
    const focus = data?.focus_point;
    const pathPoints = data?.path_points ?? [];

    const renderVectors = () => (
      <>
        {vectors.map((vector: OverlayVector, index: number) => (
          <React.Fragment key={index}>
            <Line
              x1={toPxX(vector.start[0], width)}
              y1={toPxY(vector.start[1], height)}
              x2={toPxX(vector.end[0], width)}
              y2={toPxY(vector.end[1], height)}
              stroke={vector.color}
              strokeWidth={4}
            />
            {vector.label ? (
              <SvgText
                x={toPxX(vector.end[0], width) + 6}
                y={toPxY(vector.end[1], height) - 6}
                fill={vector.color}
                fontSize={14}
                fontWeight="600"
              >
                {vector.label}
              </SvgText>
            ) : null}
          </React.Fragment>
        ))}
      </>
    );

    const renderAngleArc = () => {
      if (vectors.length < 2) return null;

      const v1 = vectors[0];
      const v2 = vectors[1];
      const pivot01 = v1.start;

      const cx = toPxX(pivot01[0], width);
      const cy = toPxY(pivot01[1], height);
      const a1 = Math.atan2(v1.end[1] - pivot01[1], v1.end[0] - pivot01[0]);
      const a2 = Math.atan2(v2.end[1] - pivot01[1], v2.end[0] - pivot01[0]);

      const r = Math.max(18, Math.min(width, height) * 0.08);
      const d = arcPath(cx, cy, r, a1, a2);

      return (
        <>
          <Circle cx={cx} cy={cy} r={5} fill="#fff" opacity={0.9} />
          <Path d={d} stroke="#fff" strokeWidth={3} fill="none" opacity={0.9} />
        </>
      );
    };

    const renderPositionMarker = () => {
      if (!focus) return null;
      const cx = toPxX(focus.x, width);
      const cy = toPxY(focus.y, height);
      return (
        <>
          <Circle cx={cx} cy={cy} r={10} stroke="#fff" strokeWidth={3} fill="none" opacity={0.95} />
          <Circle cx={cx} cy={cy} r={4} fill="#fff" opacity={0.95} />
        </>
      );
    };

    const renderPathTrace = () => {
      if (pathPoints.length < 2) return null;
      const d = pathPoints
        .map(([x01, y01], i) => {
          const x = toPxX(x01, width);
          const y = toPxY(y01, height);
          return `${i === 0 ? 'M' : 'L'} ${x} ${y}`;
        })
        .join(' ');
      return <Path d={d} stroke="#00ff88" strokeWidth={4} fill="none" opacity={0.95} />;
    };

    const renderOverlay = () => {
      if (overlayType === 'POSITION_MARKER') return renderPositionMarker();
      if (overlayType === 'PATH_TRACE') return renderPathTrace();
      // default ANGLE_CORRECTION
      return (
        <>
          {renderVectors()}
          {renderAngleArc()}
        </>
      );
    };

    return (
        // pointerEvents="none" ensures taps on the SVG pass through to the 
        // video player controls (or the "Continue" button built by Dev 1)
        <View style={[styles.overlayContainer, { height }]} pointerEvents="none">
 
        <Svg width={width} height={height}>
          {renderOverlay()}
        </Svg>
      

        </View>
    );
}

const styles = StyleSheet.create({
    overlayContainer: {
        ...StyleSheet.absoluteFillObject,
        // Do not add background color! It must be completely transparent 
        // so the paused video behind it remains visible.
    }
});
