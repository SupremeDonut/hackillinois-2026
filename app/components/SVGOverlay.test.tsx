/**
 * Unit tests for SVGOverlay.
 * Run with: npx expo test or npm test (after adding jest-expo and @testing-library/react-native).
 */
import React from 'react';
import { render } from '@testing-library/react-native';
import { Dimensions } from 'react-native';
import SVGOverlay from './SVGOverlay';

const MOCK_VISUALS = {
  focus_point: { x: 0.5, y: 0.3 },
  overlay_type: 'ANGLE_CORRECTION',
  vectors: [
    { start: [0.4, 0.4] as [number, number], end: [0.5, 0.5] as [number, number], color: 'red', label: 'Current' },
    { start: [0.4, 0.4] as [number, number], end: [0.6, 0.3] as [number, number], color: 'green', label: 'Target' },
  ],
};

beforeEach(() => {
  Dimensions.get = jest.fn().mockReturnValue({ width: 400, height: 800 });
});

describe('SVGOverlay', () => {
  it('renders without crashing when given valid data', () => {
    const { toJSON } = render(<SVGOverlay data={MOCK_VISUALS} />);
    expect(toJSON()).toBeTruthy();
  });

  it('renders when data has empty vectors', () => {
    const empty = { ...MOCK_VISUALS, vectors: [] };
    const { toJSON } = render(<SVGOverlay data={empty} />);
    expect(toJSON()).toBeTruthy();
  });
});
