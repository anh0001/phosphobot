import { describe, expect, it } from "vitest";

import {
  buildRelativeTeleopCommand,
  createDefaultCalibrationProfile,
  normalizeGamepadState,
  processAnalogSticks,
  type GamepadCalibrationProfiles,
} from "./GamepadUtils";

const createGamepad = ({
  id = "Test Controller",
  mapping = "standard",
  axes = [],
  buttonValues = [],
}: {
  id?: string;
  mapping?: string;
  axes?: number[];
  buttonValues?: number[];
}): Gamepad => {
  const buttons = Array.from({ length: Math.max(8, buttonValues.length) }, (_, index) => ({
    pressed: (buttonValues[index] ?? 0) > 0,
    touched: (buttonValues[index] ?? 0) > 0,
    value: buttonValues[index] ?? 0,
  }));

  return ({
    id,
    index: 0,
    connected: true,
    mapping,
    timestamp: 0,
    axes,
    buttons,
    vibrationActuator: null,
    hapticActuators: [],
  } as unknown) as Gamepad;
};

describe("processAnalogSticks", () => {
  it("maps standard left-stick horizontal input to opposite robot y directions", () => {
    const leftPush = processAnalogSticks(
      createGamepad({ axes: [-0.8, 0, 0, 0, 0, 0, 0, -1] }),
    );
    const rightPush = processAnalogSticks(
      createGamepad({ axes: [0.8, 0, 0, 0, 0, 0, 0, -1] }),
    );

    expect(leftPush.y).toBeCloseTo(0.8);
    expect(rightPush.y).toBeCloseTo(-0.8);
  });

  it("applies the analog deadzone before generating movement", () => {
    const movement = processAnalogSticks(
      createGamepad({ axes: [0.1, -0.14, 0.12, -0.1, 0, 0, 0, -1] }),
    );

    expect(movement.x).toBeCloseTo(0);
    expect(movement.y).toBeCloseTo(0);
    expect(movement.z).toBeCloseTo(0);
    expect(movement.rz).toBeCloseTo(0);
  });

  it("uses a calibrated non-standard profile to normalize stick axes", () => {
    const controllerId = "Custom Controller";
    const profile = createDefaultCalibrationProfile(controllerId, "xinput");
    profile.calibrated = true;
    profile.createdFromStandardLayout = false;
    profile.leftStickX = { axisIndex: 4, sign: -1 };
    profile.leftStickY = { axisIndex: 5, sign: -1 };

    const calibrationProfiles: GamepadCalibrationProfiles = {
      [controllerId]: profile,
    };

    const leftPush = processAnalogSticks(
      createGamepad({
        id: controllerId,
        mapping: "xinput",
        axes: [0, 0, 0, 0, 0.75, 0, 0, -1],
      }),
      0,
      calibrationProfiles,
    );
    const rightPush = processAnalogSticks(
      createGamepad({
        id: controllerId,
        mapping: "xinput",
        axes: [0, 0, 0, 0, -0.75, 0, 0, -1],
      }),
      0,
      calibrationProfiles,
    );

    expect(leftPush.y).toBeCloseTo(0.75);
    expect(rightPush.y).toBeCloseTo(-0.75);
  });

  it("blocks uncalibrated non-standard controllers from producing stick movement", () => {
    const normalized = normalizeGamepadState(
      createGamepad({
        id: "Unknown Pad",
        mapping: "xinput",
        axes: [0.9, 0.9, 0, 0, 0, 0, 0, -1],
      }),
    );

    expect(normalized.requiresCalibration).toBe(true);
    expect(normalized.leftStickX).toBe(0);
    expect(normalized.leftStickY).toBe(0);
  });
});

describe("buildRelativeTeleopCommand", () => {
  it("omits untouched orientation axes from the request payload", () => {
    const command = buildRelativeTeleopCommand({
      x: 0.8,
      y: -0.4,
      z: 0,
      rx: 0,
      ry: 0,
      rz: 0,
      open: 1,
    });

    expect(command).toEqual({
      x: 0.8,
      y: -0.4,
      z: 0,
      open: 1,
    });
  });

  it("preserves non-zero orientation axes", () => {
    const command = buildRelativeTeleopCommand({
      x: 0,
      y: 0,
      z: 0,
      rx: 2,
      ry: 0,
      rz: -2,
      open: 1,
    });

    expect(command).toEqual({
      x: 0,
      y: 0,
      z: 0,
      rx: 2,
      rz: -2,
      open: 1,
    });
  });
});
