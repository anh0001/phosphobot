import { LoadingPage } from "@/components/common/loading";
import { PoseControlPanel } from "@/components/common/pose-control-panel";
import { SpeedSelect } from "@/components/common/speed-select";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { fetcher } from "@/lib/utils";
import { ServerStatus } from "@/types";
import {
  ArrowDown,
  ArrowDownFromLine,
  ArrowLeft,
  ArrowRight,
  ArrowUp,
  ArrowUpFromLine,
  ChevronDown,
  ChevronUp,
  Play,
  RotateCcw,
  RotateCw,
  Space,
  Square,
} from "lucide-react";
import { useEffect, useRef, useState } from "react";
import useSWR from "swr";

export function KeyboardControl() {
  const { data: serverStatus, error: serverError } = useSWR<ServerStatus>(
    ["/status"],
    fetcher,
    {
      refreshInterval: 5000,
    },
  );

  const [isMoving, setIsMoving] = useState(false);
  const [activeKey, setActiveKey] = useState<string | null>(null);
  const [selectedRobotName, setSelectedRobotName] = useState<string | null>(
    null,
  );
  const [selectedSpeed, setSelectedSpeed] = useState<number>(0.4);

  // Refs to manage our control loop and state
  const keysPressedRef = useRef(new Set<string>());
  const intervalIdRef = useRef<NodeJS.Timeout | null>(null);
  const lastExecutionTimeRef = useRef(0);
  // open state is continuous between 0 (fully closed) and 1 (fully open)
  const openStateRef = useRef(1);

  // Time in milliseconds to go from fully open to fully closed at speed = 1.
  const FULL_TRANSITION_MS = 500;

  // Configuration constants
  const BASE_URL = `http://${window.location.hostname}:${window.location.port}/`;
  const TRANSLATION_STEP_CM = 1;
  const ROTATION_STEP_DEG = 2;
  const LOOP_INTERVAL = 10; // ms, target loop frequency
  const INSTRUCTIONS_PER_SECOND = 30; // Max command send rate
  const DEBOUNCE_INTERVAL = 1000 / INSTRUCTIONS_PER_SECOND;

  interface RobotMovement {
    x: number;
    y: number;
    z: number;
    rz: number;
    rx: number;
    ry: number;
  }

  // Mappings for keys
  const KEY_MAPPINGS: Record<string, RobotMovement> = {
    f: { x: 0, y: 0, z: TRANSLATION_STEP_CM, rz: 0, rx: 0, ry: 0 },
    v: { x: 0, y: 0, z: -TRANSLATION_STEP_CM, rz: 0, rx: 0, ry: 0 },
    ArrowUp: { x: TRANSLATION_STEP_CM, y: 0, z: 0, rz: 0, rx: 0, ry: 0 },
    ArrowDown: { x: -TRANSLATION_STEP_CM, y: 0, z: 0, rz: 0, rx: 0, ry: 0 },
    ArrowRight: {
      x: 0,
      y: -TRANSLATION_STEP_CM,
      z: 0,
      rz: 0,
      rx: 0,
      ry: 0,
    },
    ArrowLeft: {
      x: 0,
      y: TRANSLATION_STEP_CM,
      z: 0,
      rz: 0,
      rx: 0,
      ry: 0,
    },
    q: { x: 0, y: 0, z: 0, rz: ROTATION_STEP_DEG, rx: 0, ry: 0 },
    e: { x: 0, y: 0, z: 0, rz: -ROTATION_STEP_DEG, rx: 0, ry: 0 },
    d: { x: 0, y: 0, z: 0, rz: 0, rx: ROTATION_STEP_DEG, ry: 0 },
    g: { x: 0, y: 0, z: 0, rz: 0, rx: -ROTATION_STEP_DEG, ry: 0 },
    b: { x: 0, y: 0, z: 0, rz: 0, rx: 0, ry: ROTATION_STEP_DEG },
    c: { x: 0, y: 0, z: 0, rz: 0, rx: 0, ry: -ROTATION_STEP_DEG },
    " ": { x: 0, y: 0, z: 0, rz: 0, rx: 0, ry: 0 }, // Placeholder, logic handled separately
  };

  const robotIDFromName = (name?: string | null) => {
    if (name === undefined || name === null || !serverStatus?.robot_status) {
      return 0;
    }
    const index = serverStatus.robot_status.findIndex(
      (robot) => robot.device_name === name,
    );
    return index === -1 ? 0 : index;
  };

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const postData = async (url: string, data: any, queryParam?: any) => {
    try {
      let newUrl = url;
      if (queryParam) {
        const urlParams = new URLSearchParams(queryParam);
        if (urlParams.toString()) {
          newUrl += "?" + urlParams.toString();
        }
      }

      const response = await fetch(newUrl, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(data),
      });
      if (!response.ok) {
        throw new Error(`Network response was not ok: ${response.statusText}`);
      }
      return await response.json();
    } catch (error) {
      console.error("Error posting data:", error);
    }
  };

  // Effect for handling keyboard inputs and preventing page scroll
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      // 1. Prevent page scrolling from control keys when active
      if (
        isMoving &&
        ["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", " "].includes(
          event.key,
        )
      ) {
        event.preventDefault();
      }

      // 2. Gate all robot control inputs on the isMoving state
      if (!isMoving || event.repeat) return;

      const key = event.key;
      const keyLower = key.toLowerCase();

      // Check if the key is a recognized control key
      if (KEY_MAPPINGS[key] || KEY_MAPPINGS[keyLower]) {
        const effectiveKey = KEY_MAPPINGS[key] ? key : keyLower;
        setActiveKey(effectiveKey);
        keysPressedRef.current.add(effectiveKey);

        // UX Feature for Spacebar: If gripper is not fully open, a single press will fully open it.
        if (key === " " && openStateRef.current < 0.99) {
          openStateRef.current = 1;
          const data = { x: 0, y: 0, z: 0, rx: 0, ry: 0, rz: 0, open: 1 };
          postData(BASE_URL + "move/relative", data, {
            robot_id: robotIDFromName(selectedRobotName),
          });
          // Consume this press; don't add to keysPressedRef to avoid immediate closing.
          keysPressedRef.current.delete(" ");
        }
      }
    };

    const handleKeyUp = (event: KeyboardEvent) => {
      // Always clear keys on keyup to prevent "stuck" keys
      setActiveKey(null);
      const key = event.key;
      const keyLower = key.toLowerCase();
      keysPressedRef.current.delete(key);
      keysPressedRef.current.delete(keyLower);
    };

    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("keyup", handleKeyUp);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("keyup", handleKeyUp);
    };
  }, [isMoving, selectedRobotName, serverStatus]);

  // Effect for setting the default selected robot
  useEffect(() => {
    if (
      !selectedRobotName &&
      serverStatus?.robot_status &&
      serverStatus.robot_status.length > 0 &&
      serverStatus.robot_status[0].device_name
    ) {
      setSelectedRobotName(serverStatus.robot_status[0].device_name);
    }
  }, [serverStatus, selectedRobotName]);

  // The main control loop
  useEffect(() => {
    const controlRobot = () => {
      const currentTime = Date.now();
      const deltaT_ms = currentTime - lastExecutionTimeRef.current;

      if (deltaT_ms < DEBOUNCE_INTERVAL) {
        return; // Throttle command sending
      }

      // **FIX**: Only run control logic if isMoving is true.
      if (!isMoving) {
        lastExecutionTimeRef.current = currentTime; // Update time to prevent a large jump on start
        return;
      }

      const lastOpenState = openStateRef.current;
      let openStateChanged = false;

      // --- 1. Gripper State Calculation ---
      const gripperChangePerMs = (1.0 / FULL_TRANSITION_MS) * selectedSpeed;
      const gripperChangeAmount = gripperChangePerMs * deltaT_ms;

      if (keysPressedRef.current.has(" ")) {
        openStateRef.current -= gripperChangeAmount; // Close gripper
      } else {
        openStateRef.current += gripperChangeAmount; // Open gripper
      }
      openStateRef.current = Math.max(0, Math.min(1, openStateRef.current)); // Clamp

      if (Math.abs(openStateRef.current - lastOpenState) > 0.001) {
        openStateChanged = true;
      }

      // --- 2. Robot Movement Calculation ---
      let deltaX = 0,
        deltaY = 0,
        deltaZ = 0,
        deltaRZ = 0,
        deltaRX = 0,
        deltaRY = 0;
      let hasMovement = false;

      keysPressedRef.current.forEach((key) => {
        const move = KEY_MAPPINGS[key.toLowerCase()] || KEY_MAPPINGS[key];
        if (move && key !== " ") {
          deltaX += move.x;
          deltaY += move.y;
          deltaZ += move.z;
          deltaRZ += move.rz;
          deltaRX += move.rx;
          deltaRY += move.ry;
        }
      });

      if (deltaX || deltaY || deltaZ || deltaRZ || deltaRX || deltaRY) {
        hasMovement = true;
        deltaX *= selectedSpeed;
        deltaY *= selectedSpeed;
        deltaZ *= selectedSpeed;
        deltaRX *= selectedSpeed;
        deltaRY *= selectedSpeed;
        deltaRZ *= selectedSpeed;
      }

      // --- 3. Send Command if Needed ---
      if (hasMovement || openStateChanged) {
        const data = {
          x: deltaX,
          y: deltaY,
          z: deltaZ,
          rx: deltaRX,
          ry: deltaRY,
          rz: deltaRZ,
          open: openStateRef.current,
        };
        postData(BASE_URL + "move/relative", data, {
          robot_id: robotIDFromName(selectedRobotName),
        });
      }

      lastExecutionTimeRef.current = currentTime;
    };

    intervalIdRef.current = setInterval(controlRobot, LOOP_INTERVAL);

    return () => {
      if (intervalIdRef.current) {
        clearInterval(intervalIdRef.current);
      }
    };
  }, [isMoving, selectedSpeed, serverStatus, selectedRobotName]);

  const startKeyPressFromUI = (
    key: string,
    e?: React.MouseEvent | React.TouchEvent,
  ) => {
    e?.preventDefault();
    if (!isMoving) return;

    setActiveKey(key);
    const keyLower = key.toLowerCase();
    const effectiveKey = KEY_MAPPINGS[key] ? key : keyLower;

    if (key === " " && openStateRef.current < 0.99) {
      openStateRef.current = 1;
      const data = { x: 0, y: 0, z: 0, rx: 0, ry: 0, rz: 0, open: 1 };
      postData(BASE_URL + "move/relative", data, {
        robot_id: robotIDFromName(selectedRobotName),
      });
      return;
    }

    keysPressedRef.current.add(effectiveKey);
  };

  const endKeyPressFromUI = (key: string) => {
    setActiveKey(null);
    const keyLower = key.toLowerCase();
    keysPressedRef.current.delete(key);
    keysPressedRef.current.delete(keyLower);
  };

  const initRobot = async () => {
    try {
      await postData(
        BASE_URL + "move/init",
        {},
        { robot_id: robotIDFromName(selectedRobotName) },
      );
      await new Promise((resolve) => setTimeout(resolve, 2000));
      const initData = { x: 0, y: 0, z: 0, rx: 0, ry: 0, rz: 0, open: 1 };
      await postData(BASE_URL + "move/absolute", initData, {
        robot_id: robotIDFromName(selectedRobotName),
      });
      openStateRef.current = 1; // Sync local state
    } catch (error) {
      console.error("Error during init:", error);
    }
  };

  const startMoving = async () => {
    await initRobot();
    setIsMoving(true);
  };

  const stopMoving = async () => {
    setIsMoving(false);
    // **FIX**: Clear any held keys to prevent them from being "stuck"
    keysPressedRef.current.clear();
    setActiveKey(null);
  };

  const clearTeleopInputs = () => {
    keysPressedRef.current.clear();
    setActiveKey(null);
  };

  const controls = [
    {
      key: "ArrowUp",
      label: "UP",
      description: "Move forward (+X)",
      icon: <ArrowUp className="size-5" />,
    },
    {
      key: "ArrowDown",
      label: "DOWN",
      description: "Move backward (-X)",
      icon: <ArrowDown className="size-5" />,
    },
    {
      key: "ArrowLeft",
      label: "LEFT",
      description: "Move left",
      icon: <ArrowLeft className="size-5" />,
    },
    {
      key: "ArrowRight",
      label: "RIGHT",
      description: "Move right",
      icon: <ArrowRight className="size-5" />,
    },
    {
      key: "f",
      label: "F",
      description: "Increase Z (move up)",
      icon: <ChevronUp className="size-5" />,
    },
    {
      key: "v",
      label: "V",
      description: "Decrease Z (move down)",
      icon: <ChevronDown className="size-5" />,
    },
    {
      key: "q",
      label: "Q",
      description: "Yaw left",
      icon: <RotateCcw className="size-5" />,
    },
    {
      key: "e",
      label: "E",
      description: "Yaw right",
      icon: <RotateCw className="size-5" />,
    },
    {
      key: "d",
      label: "D",
      description: "Wrist pitch up",
      icon: <ArrowUpFromLine className="size-5" />,
    },
    {
      key: "g",
      label: "G",
      description: "Wrist pitch down",
      icon: <ArrowDownFromLine className="size-5" />,
    },
    {
      key: "b",
      label: "B",
      description: "Wrist roll clockwise",
      icon: <RotateCw className="size-5" />,
    },
    {
      key: "c",
      label: "C",
      description: "Wrist roll counter-clockwise",
      icon: <RotateCcw className="size-5" />,
    },
    {
      key: " ",
      label: "SPACE",
      description:
        "Hold to close gripper, release to open. Press once to fully open.",
      icon: <Space className="size-5" />,
    },
  ];

  if (serverError) return <div>Failed to load server status.</div>;
  if (!serverStatus) return <LoadingPage />;

  return (
    <div className="container mx-auto px-4 py-6 lg:flex lg:flex-row lg:gap-8 space-y-8 lg:space-y-0">
      <div className="md:w-1/2 space-y-6">
        <Card>
          <CardContent className="pt-6 flex flex-col items-center gap-2">
            <div className="flex flex-wrap items-end justify-center gap-2">
              <div className="flex flex-col gap-y-2">
                <Label>Select Robot to Control</Label>
                <Select
                  value={selectedRobotName || ""}
                  onValueChange={(value) => setSelectedRobotName(value)}
                  disabled={isMoving}
                >
                  <SelectTrigger
                    id="follower-robot"
                    className="max-w-xs truncate"
                  >
                    <SelectValue placeholder="Select robot to move" />
                  </SelectTrigger>
                  <SelectContent>
                    {serverStatus.robot_status.map((robot) => (
                      <SelectItem
                        key={robot.device_name}
                        value={robot.device_name || "Undefined port"}
                      >
                        <span className="truncate">
                          {robot.name} ({robot.device_name})
                        </span>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <SpeedSelect
                defaultValue={selectedSpeed}
                onChange={(newSpeed) => setSelectedSpeed(newSpeed)}
                title="Movement speed"
                minSpeed={0.1}
                maxSpeed={2.0}
                step={0.1}
              />
            </div>
            <div className="flex justify-center items-center gap-2">
              <Button
                variant="default"
                onClick={startMoving}
                disabled={!selectedRobotName || isMoving}
              >
                <Play className="mr-2 h-4 w-4" />
                {isMoving ? "Controlling..." : "Start Keyboard Control"}
              </Button>
              <Button
                variant="destructive"
                onClick={stopMoving}
                disabled={!isMoving}
              >
                <Square className="mr-2 h-4 w-4" />
                Stop
              </Button>
            </div>
            <div className="w-full max-w-md rounded-md border p-4 text-sm text-muted-foreground space-y-3">
              <div>
                <p className="font-medium text-foreground">Translation</p>
                <p>
                  <span className="font-mono">↑/↓</span> forward and backward,{" "}
                  <span className="font-mono">←/→</span> left and right,{" "}
                  <span className="font-mono">F/V</span> up and down.
                </p>
              </div>
              <div>
                <p className="font-medium text-foreground">Orientation</p>
                <p>
                  <span className="font-mono">Q/E</span> yaw,{" "}
                  <span className="font-mono">D/G</span> wrist pitch,{" "}
                  <span className="font-mono">B/C</span> wrist roll.
                </p>
              </div>
              <div>
                <p className="font-medium text-foreground">Gripper</p>
                <p>
                  <span className="font-mono">SPACE</span> closes while held and
                  re-opens on release.
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
        <PoseControlPanel
          robotId={robotIDFromName(selectedRobotName)}
          serverStatus={serverStatus}
          onBeforeAction={clearTeleopInputs}
        />
      </div>

      <div className="md:w-1/2">
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-3">
          {controls.map((control) => (
            <TooltipProvider key={control.key}>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Card
                    className={`flex flex-col items-center justify-center p-3 cursor-pointer select-none hover:bg-accent transition-colors ${
                      activeKey === control.key
                        ? "bg-primary text-primary-foreground"
                        : "bg-card"
                    }`}
                    onMouseDown={(e) => startKeyPressFromUI(control.key, e)}
                    onTouchStart={(e) => startKeyPressFromUI(control.key, e)}
                    onMouseUp={() => endKeyPressFromUI(control.key)}
                    onMouseLeave={() => endKeyPressFromUI(control.key)}
                    onTouchEnd={() => endKeyPressFromUI(control.key)}
                    onTouchCancel={() => endKeyPressFromUI(control.key)}
                  >
                    {control.icon}
                    <span className="mt-2 text-sm font-bold">
                      {control.label}
                    </span>
                  </Card>
                </TooltipTrigger>
                <TooltipContent>
                  <p>{control.description}</p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          ))}
        </div>
      </div>
    </div>
  );
}
