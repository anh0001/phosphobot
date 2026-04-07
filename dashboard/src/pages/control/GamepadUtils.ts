// GamepadUtils.ts - Types, constants, and utility functions

import type { ServerStatus } from "@/types";

// ==================== TYPES ====================

export interface ControllerArmPair {
  controller_index: number | null;
  robot_name: string | null;
  controller_name: string;
}

export interface MultiArmGroup {
  id: string;
  name: string;
  controller_index: number | null;
  controller_name: string;
  robot_names: string[];
  control_mode: "synchronized" | "sequential";
  active_robot_index: number; // For sequential mode
}

export interface GamepadState {
  connected: boolean;
  buttons: boolean[];
  buttonValues: number[];
  axes: number[];
}

export interface RobotMovement {
  x: number;
  y: number;
  z: number;
  rz: number;
  rx: number;
  ry: number;
  toggleOpen?: boolean;
}

export interface ControllerState {
  buttonsPressed: Set<string>;
  lastExecutionTime: number;
  openState: number;
  lastButtonStates: boolean[];
  lastTriggerValue: number;
  triggerControlActive: boolean;
  resetSent: boolean;
}

export interface AnalogValues {
  leftTrigger: number;
  rightTrigger: number;
  leftStickX: number;
  leftStickY: number;
  rightStickX: number;
  rightStickY: number;
}

export interface GamepadInfo {
  index: number;
  id: string;
  name: string;
  mapping: string;
  requiresCalibration: boolean;
  calibrated: boolean;
}

export type ControlType =
  | "analog-vertical"
  | "analog-horizontal"
  | "digital"
  | "trigger";
export type ConfigMode = "individual" | "multi-arm";

export interface Control {
  key: string;
  label: string;
  buttons: string[];
  description: string;
  icon: React.ReactNode;
  type: ControlType;
}

export type CanonicalAxisName =
  | "leftStickX"
  | "leftStickY"
  | "rightStickX"
  | "rightStickY";

export interface AxisBinding {
  axisIndex: number;
  sign: 1 | -1;
}

export interface GamepadCalibrationProfile {
  controllerId: string;
  mapping: string;
  calibrated: boolean;
  createdFromStandardLayout: boolean;
  updatedAt: number;
  leftStickX: AxisBinding;
  leftStickY: AxisBinding;
  rightStickX: AxisBinding;
  rightStickY: AxisBinding;
}

export type GamepadCalibrationProfiles = Record<
  string,
  GamepadCalibrationProfile
>;

export type CalibrationStepId =
  | "left-stick-left"
  | "left-stick-right"
  | "left-stick-up"
  | "left-stick-down"
  | "right-stick-left"
  | "right-stick-right"
  | "right-stick-up"
  | "right-stick-down";

export interface CalibrationAxisSample {
  axisIndex: number;
  sign: 1 | -1;
  magnitude: number;
}

export type CalibrationSamples = Partial<
  Record<CalibrationStepId, CalibrationAxisSample>
>;

export interface CalibrationStepDefinition {
  id: CalibrationStepId;
  label: string;
  description: string;
}

export interface ControllerCalibrationStatus {
  profile: GamepadCalibrationProfile;
  requiresCalibration: boolean;
  calibrated: boolean;
  isStandardMapping: boolean;
}

export interface NormalizedGamepadState extends AnalogValues {
  rawAxes: number[];
  rawButtonValues: number[];
  mapping: string;
  calibrated: boolean;
  requiresCalibration: boolean;
  isStandardMapping: boolean;
  profile: GamepadCalibrationProfile;
}

export interface RelativeTeleopCommand {
  x: number;
  y: number;
  z: number;
  rx?: number;
  ry?: number;
  rz?: number;
  open: number;
}

// ==================== CONSTANTS ====================

// Configuration constants
const isBrowser = typeof window !== "undefined";

export const BASE_URL = isBrowser
  ? `http://${window.location.hostname}:${window.location.port}/`
  : "/";
export const STEP_SIZE = 1; // Translation step in centimeters
export const ROTATION_STEP_DEG = 2; // Rotation step in degrees
export const LOOP_INTERVAL = 10; // ms (~100 Hz)
export const INSTRUCTIONS_PER_SECOND = 30;
export const DEBOUNCE_INTERVAL = 1000 / INSTRUCTIONS_PER_SECOND;
export const AXIS_DEADZONE = 0.15; // Deadzone for analog sticks
export const AXIS_SCALE = 1; // Scale factor for analog stick movement
export const CALIBRATION_ACTIVATION_THRESHOLD = 0.45;
export const GAMEPAD_CALIBRATION_STORAGE_KEY = "gamepad-calibration-profiles";

export const CALIBRATION_STEP_ORDER: CalibrationStepDefinition[] = [
  {
    id: "left-stick-left",
    label: "Left Stick Left",
    description: "Push the left stick fully left and hold for a moment.",
  },
  {
    id: "left-stick-right",
    label: "Left Stick Right",
    description: "Push the left stick fully right and hold for a moment.",
  },
  {
    id: "left-stick-up",
    label: "Left Stick Up",
    description: "Push the left stick fully up and hold for a moment.",
  },
  {
    id: "left-stick-down",
    label: "Left Stick Down",
    description: "Pull the left stick fully down and hold for a moment.",
  },
  {
    id: "right-stick-left",
    label: "Right Stick Left",
    description: "Push the right stick fully left and hold for a moment.",
  },
  {
    id: "right-stick-right",
    label: "Right Stick Right",
    description: "Push the right stick fully right and hold for a moment.",
  },
  {
    id: "right-stick-up",
    label: "Right Stick Up",
    description: "Push the right stick fully up and hold for a moment.",
  },
  {
    id: "right-stick-down",
    label: "Right Stick Down",
    description: "Pull the right stick fully down and hold for a moment.",
  },
];

const STANDARD_MAPPING = "standard";

// Gamepad button mappings (standard gamepad layout)
export const BUTTON_MAPPINGS: Record<number, RobotMovement> = {
  12: { x: 0, y: 0, z: 0, rz: 0, rx: ROTATION_STEP_DEG, ry: 0 }, // D-pad up - wrist pitch up
  13: { x: 0, y: 0, z: 0, rz: 0, rx: -ROTATION_STEP_DEG, ry: 0 }, // D-pad down - wrist pitch down
  14: { x: 0, y: 0, z: 0, rz: 0, rx: 0, ry: -ROTATION_STEP_DEG }, // D-pad left - wrist roll counter-clockwise
  15: { x: 0, y: 0, z: 0, rz: 0, rx: 0, ry: ROTATION_STEP_DEG }, // D-pad right - wrist roll clockwise
  4: { x: 0, y: 0, z: 0, rz: 0, rx: 0, ry: 0, toggleOpen: true }, // L1/LB - toggle gripper
  5: { x: 0, y: 0, z: 0, rz: 0, rx: 0, ry: 0, toggleOpen: true }, // R1/RB - toggle gripper
  0: { x: 0, y: 0, z: 0, rz: 0, rx: -ROTATION_STEP_DEG, ry: 0 }, // A/X button - wrist pitch down
  1: { x: 0, y: 0, z: 0, rz: 0, rx: 0, ry: ROTATION_STEP_DEG }, // B/Circle - wrist roll clockwise
  2: { x: 0, y: 0, z: 0, rz: 0, rx: 0, ry: -ROTATION_STEP_DEG }, // X/Square - wrist roll counter-clockwise
  3: { x: 0, y: 0, z: 0, rz: 0, rx: ROTATION_STEP_DEG, ry: 0 }, // Y/Triangle - wrist pitch up
  9: { x: 0, y: 0, z: 0, rz: 0, rx: 0, ry: 0 }, // Start/Menu - move to sleep position
  10: { x: 0, y: 0, z: 0, rz: 0, rx: 0, ry: 0 }, // Start/Menu (alternate index) - move to sleep position
};

// Special button mappings for multi-arm control
export const MULTI_ARM_SPECIAL_BUTTONS: Record<number, string> = {
  8: "switch_robot", // Select button (8) or Back button for switching active robot in sequential mode
  10: "mode_switch", // L3/R3 (stick clicks) for mode switching
  11: "mode_switch",
};

// Button names for display
export const BUTTON_NAMES = [
  "A/X",
  "B/Circle",
  "X/Square",
  "Y/Triangle",
  "L1/LB",
  "R1/RB",
  "L2/LT",
  "R2/RT",
  "Select/Back",
  "Start/Menu",
  "L3",
  "R3",
  "D-Pad Up",
  "D-Pad Down",
  "D-Pad Left",
  "D-Pad Right",
  "Home/Guide",
];

// ==================== NORMALIZATION HELPERS ====================

const clampUnit = (value: number): number => {
  return Math.max(-1, Math.min(1, value));
};

export const applyAxisDeadzone = (value: number): number => {
  return Math.abs(value) > AXIS_DEADZONE ? value : 0;
};

export const createDefaultCalibrationProfile = (
  controllerId: string,
  mapping: string = STANDARD_MAPPING,
): GamepadCalibrationProfile => {
  const isStandardMapping = mapping === STANDARD_MAPPING;

  return {
    controllerId,
    mapping,
    calibrated: isStandardMapping,
    createdFromStandardLayout: true,
    updatedAt: Date.now(),
    leftStickX: { axisIndex: 0, sign: 1 },
    leftStickY: { axisIndex: 1, sign: 1 },
    rightStickX: { axisIndex: 2, sign: 1 },
    rightStickY: { axisIndex: 3, sign: 1 },
  };
};

export const getControllerCalibrationStatus = (
  gamepad: Pick<Gamepad, "id" | "mapping">,
  profiles: GamepadCalibrationProfiles = {},
): ControllerCalibrationStatus => {
  const isStandardMapping = gamepad.mapping === STANDARD_MAPPING;
  const defaultProfile = createDefaultCalibrationProfile(
    gamepad.id,
    gamepad.mapping,
  );
  const profile = profiles[gamepad.id] ?? defaultProfile;
  const calibrated = isStandardMapping ? true : profile.calibrated;

  return {
    profile: {
      ...profile,
      mapping: gamepad.mapping,
      calibrated,
    },
    calibrated,
    isStandardMapping,
    requiresCalibration: !isStandardMapping && !calibrated,
  };
};

export const getNormalizedAxisValue = (
  axes: readonly number[],
  binding: AxisBinding,
): number => {
  const rawValue = axes[binding.axisIndex] ?? 0;
  return clampUnit(rawValue * binding.sign);
};

const getTriggerValues = (gamepad: Pick<Gamepad, "axes" | "buttons">) => {
  let leftTrigger = 0;
  let rightTrigger = 0;

  if (gamepad.axes.length >= 8) {
    leftTrigger = gamepad.axes[6] > 0.1 ? gamepad.axes[6] : 0;
    rightTrigger = gamepad.axes[7] > -0.9 ? (gamepad.axes[7] + 1) / 2 : 0;
  }

  if (leftTrigger === 0 && gamepad.buttons.length > 6 && gamepad.buttons[6]) {
    leftTrigger = gamepad.buttons[6].value || (gamepad.buttons[6].pressed ? 1 : 0);
  }

  if (rightTrigger === 0 && gamepad.buttons.length > 7 && gamepad.buttons[7]) {
    rightTrigger = gamepad.buttons[7].value || (gamepad.buttons[7].pressed ? 1 : 0);
  }

  return {
    leftTrigger: Math.max(0, Math.min(1, leftTrigger)),
    rightTrigger: Math.max(0, Math.min(1, rightTrigger)),
  };
};

export const normalizeGamepadState = (
  gamepad: Pick<Gamepad, "id" | "mapping" | "axes" | "buttons">,
  profiles: GamepadCalibrationProfiles = {},
): NormalizedGamepadState => {
  const status = getControllerCalibrationStatus(gamepad, profiles);
  const rawAxes = Array.from(gamepad.axes);
  const rawButtonValues = Array.from(gamepad.buttons).map((button) => button.value);
  const triggers = getTriggerValues(gamepad);

  if (status.requiresCalibration) {
    return {
      leftTrigger: triggers.leftTrigger,
      rightTrigger: triggers.rightTrigger,
      leftStickX: 0,
      leftStickY: 0,
      rightStickX: 0,
      rightStickY: 0,
      rawAxes,
      rawButtonValues,
      mapping: gamepad.mapping,
      calibrated: false,
      requiresCalibration: true,
      isStandardMapping: false,
      profile: status.profile,
    };
  }

  const leftStickX = applyAxisDeadzone(
    getNormalizedAxisValue(gamepad.axes, status.profile.leftStickX),
  );
  const leftStickY = applyAxisDeadzone(
    getNormalizedAxisValue(gamepad.axes, status.profile.leftStickY),
  );
  const rightStickX = applyAxisDeadzone(
    getNormalizedAxisValue(gamepad.axes, status.profile.rightStickX),
  );
  const rightStickY = applyAxisDeadzone(
    getNormalizedAxisValue(gamepad.axes, status.profile.rightStickY),
  );

  return {
    leftTrigger: triggers.leftTrigger,
    rightTrigger: triggers.rightTrigger,
    leftStickX,
    leftStickY,
    rightStickX,
    rightStickY,
    rawAxes,
    rawButtonValues,
    mapping: gamepad.mapping,
    calibrated: status.calibrated,
    requiresCalibration: false,
    isStandardMapping: status.isStandardMapping,
    profile: status.profile,
  };
};

export const detectDominantAxisSample = (
  axes: readonly number[],
  baseline: readonly number[],
  threshold: number = CALIBRATION_ACTIVATION_THRESHOLD,
): CalibrationAxisSample | null => {
  let axisIndex = -1;
  let maxMagnitude = 0;
  let signedDelta = 0;

  for (let index = 0; index < axes.length; index += 1) {
    const delta = (axes[index] ?? 0) - (baseline[index] ?? 0);
    const magnitude = Math.abs(delta);
    if (magnitude > maxMagnitude) {
      maxMagnitude = magnitude;
      axisIndex = index;
      signedDelta = delta;
    }
  }

  if (axisIndex === -1 || maxMagnitude < threshold || signedDelta === 0) {
    return null;
  }

  const sample = {
    axisIndex,
    sign: (signedDelta > 0 ? 1 : -1) as 1 | -1,
    magnitude: maxMagnitude,
  };

  return sample;
};

const assertOpposedAxisSamples = (
  negativeSample: CalibrationAxisSample | undefined,
  positiveSample: CalibrationAxisSample | undefined,
  axisName: CanonicalAxisName,
): AxisBinding => {
  if (!negativeSample || !positiveSample) {
    throw new Error(`Missing calibration sample for ${axisName}.`);
  }

  if (negativeSample.axisIndex !== positiveSample.axisIndex) {
    throw new Error(
      `${axisName} was detected on different raw axes (${negativeSample.axisIndex} and ${positiveSample.axisIndex}).`,
    );
  }

  if (negativeSample.sign === positiveSample.sign) {
    throw new Error(
      `${axisName} positive and negative directions used the same raw sign. Re-run calibration.`,
    );
  }

  return {
    axisIndex: positiveSample.axisIndex,
    sign: positiveSample.sign,
  };
};

export const deriveCalibrationProfileFromSamples = (
  controllerId: string,
  mapping: string,
  samples: CalibrationSamples,
): GamepadCalibrationProfile => {
  const leftStickX = assertOpposedAxisSamples(
    samples["left-stick-left"],
    samples["left-stick-right"],
    "leftStickX",
  );
  const leftStickY = assertOpposedAxisSamples(
    samples["left-stick-up"],
    samples["left-stick-down"],
    "leftStickY",
  );
  const rightStickX = assertOpposedAxisSamples(
    samples["right-stick-left"],
    samples["right-stick-right"],
    "rightStickX",
  );
  const rightStickY = assertOpposedAxisSamples(
    samples["right-stick-up"],
    samples["right-stick-down"],
    "rightStickY",
  );

  return {
    controllerId,
    mapping,
    calibrated: true,
    createdFromStandardLayout: false,
    updatedAt: Date.now(),
    leftStickX,
    leftStickY,
    rightStickX,
    rightStickY,
  };
};

// ==================== UTILITY FUNCTIONS ====================

export const robotIDFromName = (
  name?: string | null,
  serverStatus?: ServerStatus,
): number => {
  if (name === undefined || name === null || !serverStatus?.robot_status) {
    return 0;
  }
  const index = serverStatus.robot_status.findIndex(
    (robot) => robot.device_name === name,
  );
  return index === -1 ? 0 : index;
};

export const postData = async (
  url: string,
  data: Record<string, unknown>,
  queryParam?: Record<string, string | number>,
) => {
  try {
    const sanitizedData = Object.fromEntries(
      Object.entries(data).filter(([, value]) => value !== undefined),
    );

    let newUrl = url;
    if (queryParam) {
      const urlParams = new URLSearchParams();
      Object.entries(queryParam).forEach(([key, value]) => {
        urlParams.append(key, value.toString());
      });
      if (urlParams.toString()) {
        newUrl += "?" + urlParams.toString();
      }
    }

    const response = await fetch(newUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(sanitizedData),
    });
    if (!response.ok) {
      throw new Error(`Network response was not ok: ${response.statusText}`);
    }
    return await response.json();
  } catch (error) {
    console.error("Error posting data:", error);
  }
};

export const processAnalogSticks = (
  gamepad: Gamepad,
  lastTriggerValue: number = 0,
  profiles: GamepadCalibrationProfiles = {},
): RobotMovement & { gripperValue?: number; requiresCalibration?: boolean } => {
  const movement: RobotMovement & {
    gripperValue?: number;
    requiresCalibration?: boolean;
  } = {
    x: 0,
    y: 0,
    z: 0,
    rz: 0,
    rx: 0,
    ry: 0,
  };

  const normalized = normalizeGamepadState(gamepad, profiles);

  if (normalized.requiresCalibration) {
    movement.requiresCalibration = true;
    return movement;
  }

  movement.x = -normalized.leftStickY * STEP_SIZE * AXIS_SCALE; // Forward/backward
  movement.y = -normalized.leftStickX * STEP_SIZE * AXIS_SCALE; // Left/right strafe
  movement.rz = normalized.rightStickX * ROTATION_STEP_DEG * AXIS_SCALE; // Yaw
  movement.z = -normalized.rightStickY * STEP_SIZE * AXIS_SCALE; // Up/down

  const triggerValue = Math.max(
    normalized.leftTrigger,
    normalized.rightTrigger,
  );

  if (triggerValue > 0 || lastTriggerValue > 0) {
    movement.gripperValue = triggerValue;
  }

  return movement;
};

export const getControlName = (index: number): string => {
  if (index === 0) return "wrist-pitch-down";
  else if (index === 1) return "wrist-roll-right";
  else if (index === 2) return "wrist-roll-left";
  else if (index === 3) return "wrist-pitch-up";
  else if (index === 4 || index === 5) return "gripper-toggle";
  else if (index === 9 || index === 10) return "sleep";
  else if (index === 12) return "wrist-pitch-up";
  else if (index === 13) return "wrist-pitch-down";
  else if (index === 14) return "wrist-roll-left";
  else if (index === 15) return "wrist-roll-right";

  return BUTTON_NAMES[index] || `Button ${index}`;
};

export const getRobotsToControl = (
  config: ControllerArmPair | MultiArmGroup,
  configMode: ConfigMode,
): string[] => {
  if (configMode === "individual") {
    return "robot_name" in config && config.robot_name ? [config.robot_name] : [];
  } else {
    // Multi-arm group
    if (!("control_mode" in config)) {
      return [];
    }

    if (config.control_mode === "sequential") {
      return [config.robot_names[config.active_robot_index]];
    } else {
      return config.robot_names;
    }
  }
};

export const applyControlMode = (
  data: RelativeTeleopCommand,
  controlMode: string,
): RelativeTeleopCommand => {
  switch (controlMode) {
    case "synchronized":
      return data; // All robots move identically
    case "sequential":
      return data; // Only active robot moves (handled in getRobotsToControl)
    default:
      return data;
  }
};

export const buildRelativeTeleopCommand = ({
  x,
  y,
  z,
  rx,
  ry,
  rz,
  open,
}: {
  x: number;
  y: number;
  z: number;
  rx: number;
  ry: number;
  rz: number;
  open: number;
}): RelativeTeleopCommand => {
  return {
    x,
    y,
    z,
    rx: rx === 0 ? undefined : rx,
    ry: ry === 0 ? undefined : ry,
    rz: rz === 0 ? undefined : rz,
    open,
  };
};

export const initRobot = async (
  robotName: string,
  serverStatus?: ServerStatus,
) => {
  try {
    await postData(
      BASE_URL + "move/init",
      {},
      {
        robot_id: robotIDFromName(robotName, serverStatus),
      },
    );
    await new Promise((resolve) => setTimeout(resolve, 2000));
    const initData = {
      x: 0,
      y: 0,
      z: 0,
      rx: 0,
      ry: 0,
      rz: 0,
      open: 1,
    };
    await postData(BASE_URL + "move/absolute", initData, {
      robot_id: robotIDFromName(robotName, serverStatus),
    });
  } catch (error) {
    console.error("Error during init:", error);
  }
};

export const getAvailableGamepads = (
  profiles: GamepadCalibrationProfiles = {},
): GamepadInfo[] => {
  const gamepads = navigator.getGamepads();
  const available: GamepadInfo[] = [];

  for (let i = 0; i < gamepads.length; i += 1) {
    const gamepad = gamepads[i];
    if (gamepad) {
      const fullName = gamepad.id;
      const shortName = fullName.split(" ").slice(0, 3).join(" ");
      const calibrationStatus = getControllerCalibrationStatus(gamepad, profiles);

      available.push({
        index: i,
        id: fullName,
        name: shortName || `Controller ${i + 1}`,
        mapping: gamepad.mapping || "unknown",
        requiresCalibration: calibrationStatus.requiresCalibration,
        calibrated: calibrationStatus.calibrated,
      });
    }
  }

  return available;
};

export const getAvailableControllers = (
  currentPairIndex: number,
  controllerArmPairs: ControllerArmPair[],
  availableGamepads: GamepadInfo[],
): GamepadInfo[] => {
  const usedControllerIds = new Set<number>();

  controllerArmPairs.forEach((pair, index) => {
    if (index !== currentPairIndex && pair.controller_index !== null) {
      usedControllerIds.add(pair.controller_index);
    }
  });

  return availableGamepads.filter(
    (gamepad) => !usedControllerIds.has(gamepad.index),
  );
};

export const getAvailableRobots = (
  currentPairIndex: number,
  controllerArmPairs: ControllerArmPair[],
  serverStatus?: ServerStatus,
) => {
  const usedRobotNames = new Set<string>();

  controllerArmPairs.forEach((pair, index) => {
    if (index !== currentPairIndex && pair.robot_name !== null) {
      usedRobotNames.add(pair.robot_name);
    }
  });

  return (
    serverStatus?.robot_status?.filter(
      (robot) => robot.device_name && !usedRobotNames.has(robot.device_name),
    ) || []
  );
};

export const getAvailableControllersForGroup = (
  availableGamepads: GamepadInfo[],
): GamepadInfo[] => {
  // Allow any controller to be used - don't exclude based on other groups
  // This enables one controller to control multiple groups
  return availableGamepads;
};

export const getAvailableRobotsForGroup = (
  currentGroupId: string,
  multiArmGroups: MultiArmGroup[],
  serverStatus?: ServerStatus,
) => {
  const usedRobotNames = new Set<string>();

  multiArmGroups.forEach((group) => {
    if (group.id !== currentGroupId) {
      group.robot_names.forEach((name) => usedRobotNames.add(name));
    }
  });

  return (
    serverStatus?.robot_status?.filter(
      (robot) => robot.device_name && !usedRobotNames.has(robot.device_name),
    ) || []
  );
};

export const extractAnalogValues = (
  gamepad: Gamepad,
  profiles: GamepadCalibrationProfiles = {},
): AnalogValues => {
  const normalized = normalizeGamepadState(gamepad, profiles);

  return {
    leftTrigger: normalized.leftTrigger,
    rightTrigger: normalized.rightTrigger,
    leftStickX: normalized.leftStickX,
    leftStickY: normalized.leftStickY,
    rightStickX: normalized.rightStickX,
    rightStickY: normalized.rightStickY,
  };
};
