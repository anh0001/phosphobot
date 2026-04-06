import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import type { ServerStatus } from "@/types";
import { Home, Save, Target } from "lucide-react";
import { useState } from "react";
import { toast } from "sonner";

const BASE_URL = `http://${window.location.hostname}:${window.location.port}/`;

interface PoseControlPanelProps {
  robotId: number;
  serverStatus?: ServerStatus;
  disabled?: boolean;
  onBeforeAction?: () => void;
}

export function PoseControlPanel({
  robotId,
  serverStatus,
  disabled = false,
  onBeforeAction,
}: PoseControlPanelProps) {
  const [loading, setLoading] = useState<string | null>(null);
  const isRecording = serverStatus?.is_recording || false;
  const isDisabled = disabled || isRecording;

  const robot = serverStatus?.robot_status?.[robotId];
  const isPiper = robot?.name === "agilex-piper";

  // Only show for Piper robots (v1)
  if (!isPiper) return null;

  const postAction = async (endpoint: string, label: string) => {
    onBeforeAction?.();
    setLoading(endpoint);
    try {
      const resp = await fetch(
        `${BASE_URL}${endpoint}?robot_id=${robotId}`,
        { method: "POST" },
      );
      if (!resp.ok) {
        const body = await resp.json().catch(() => ({}));
        throw new Error(body.detail || resp.statusText);
      }
      toast.success(`${label} completed`);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err);
      toast.error(`${label} failed: ${message}`);
    } finally {
      setLoading(null);
    }
  };

  return (
    <Card>
      <CardContent className="pt-4">
        <p className="text-sm font-medium mb-3">Pose Control</p>
        <div className="flex flex-wrap gap-2">
          <Button
            variant="outline"
            size="sm"
            disabled={isDisabled || loading !== null}
            onClick={() => postAction("move/home", "Home Pose")}
          >
            <Home className="mr-1 h-4 w-4" />
            {loading === "move/home" ? "Moving..." : "Home Pose"}
          </Button>
          <Button
            variant="outline"
            size="sm"
            disabled={isDisabled || loading !== null}
            onClick={() => postAction("move/ready", "Ready Pose")}
          >
            <Target className="mr-1 h-4 w-4" />
            {loading === "move/ready" ? "Moving..." : "Ready Pose"}
          </Button>
          <Button
            variant="outline"
            size="sm"
            disabled={isDisabled || loading !== null}
            onClick={() =>
              postAction("robot/config/ready-pose", "Save Current as Ready")
            }
          >
            <Save className="mr-1 h-4 w-4" />
            {loading === "robot/config/ready-pose"
              ? "Saving..."
              : "Save Current as Ready"}
          </Button>
        </div>
        {isRecording && (
          <p className="text-xs text-muted-foreground mt-2">
            Pose actions are disabled during recording.
          </p>
        )}
      </CardContent>
    </Card>
  );
}
