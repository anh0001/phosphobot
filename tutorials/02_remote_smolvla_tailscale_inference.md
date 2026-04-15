# Run SmolVLA Inference Remotely Over Tailscale

This tutorial shows how to run a **SmolVLA inference server** on a **remote GPU machine** and connect to it from a **Jetson robot machine** running `phosphobot`.

The setup is:

- remote GPU machine runs `phosphobot serve-smolvla`
- Jetson runs the normal `phosphobot` app
- both machines are connected through **Tailscale**
- the dashboard `Start AI control` button calls the remote inference server

This flow is for **SmolVLA** remote inference.

## What this tutorial assumes

- You already have this repo on both machines.
- Both machines are on Linux.
- Both machines can join the same Tailscale tailnet.
- The remote machine has a usable GPU for SmolVLA inference.
- You want the Jetson to keep camera capture, robot control, and AI control logic locally.

## Machine roles

Use two machines:

1. **Remote GPU machine**
   Runs the SmolVLA inference HTTP server.

2. **Jetson / robot machine**
   Runs `phosphobot`, opens the dashboard, and sends inference requests to the remote machine.

## 1. Set up the remote GPU machine

Clone the repo and enter it:

```bash
git clone https://github.com/phospho-app/phosphobot.git
cd phosphobot
```

Create or sync the Python environment:

```bash
uv sync
```

If you use a private Hugging Face model, log in first:

```bash
uv run hf auth login
```

## 2. Join Tailscale on the remote machine

Bring Tailscale up:

```bash
sudo tailscale up
```

Check the Tailscale IPv4 address:

```bash
tailscale ip -4
```

You will use that IP later from the Jetson.

Example:

```bash
100.64.0.10
```

## 3. Start the SmolVLA inference server on the remote machine

Run:

```bash
cd phosphobot
uv run --python 3.10 phosphobot serve-smolvla \
  --model-id your-org/your-smolvla-model \
  --host 0.0.0.0 \
  --port 8080
```

Replace `your-org/your-smolvla-model` with the actual model ID or local model path.

Examples:

```bash
uv run --python 3.10 phosphobot serve-smolvla \
  --model-id myname/piper-smolvla \
  --host 0.0.0.0 \
  --port 8080
```

```bash
uv run --python 3.10 phosphobot serve-smolvla \
  --model-id /home/user/models/piper_smolvla \
  --host 0.0.0.0 \
  --port 8080
```

Keep this process running.

The server exposes:

- `GET /health`
- `POST /act`

## 4. Verify the remote server locally

On the remote machine, check that the server responds:

```bash
curl http://127.0.0.1:8080/health
```

Expected result:

```json
{"status":"ok"}
```

If this does not work, fix the remote server first before touching the Jetson.

## 5. Set up the Jetson / robot machine

Clone the repo if needed:

```bash
git clone https://github.com/phospho-app/phosphobot.git
cd phosphobot
```

Start Tailscale:

```bash
sudo tailscale up
```

Check Jetson Tailscale connectivity:

```bash
tailscale status
```

You should be able to see the remote GPU machine in the same tailnet.

## 6. Test Jetson-to-remote connectivity

From the Jetson, call the remote server health endpoint:

```bash
curl http://100.64.0.10:8080/health
```

Replace `100.64.0.10` with the real Tailscale IP of the remote GPU machine.

You can also use a MagicDNS hostname if available:

```bash
curl http://gpu-box.tailnet.ts.net:8080/health
```

If this fails, `Start AI control` will fail too.

## 7. Start phosphobot on the Jetson

Run your normal robot command on the Jetson.

Example for Piper:

```bash
make CAN_INTERFACE=can_piper
```

Or run the backend directly:

```bash
cd phosphobot
uv run --python 3.10 phosphobot run --simulation=headless --can-interface can_piper
```

Open the dashboard in your browser.

## 8. Configure the dashboard to use the remote server

In the dashboard:

1. Open **Admin Settings**
2. Find **AI Inference Settings**
3. Set **Inference Mode** to `Remote URL (Tailscale)`
4. Set **Remote Inference URL** to the remote machine address

Example:

```txt
http://100.64.0.10:8080
```

or:

```txt
http://gpu-box.tailnet.ts.net:8080
```

5. Save the settings

## 9. Start AI control

In the **AI Control** page:

1. Select model type `SmolVLA`
2. Enter the same model ID that the remote server loaded
3. Enter your prompt
4. Click `Start AI control`

The Jetson should now call the remote SmolVLA server over Tailscale.

## 10. Recommended startup order

Use this order every time:

1. Start Tailscale on the remote GPU machine
2. Start `phosphobot serve-smolvla` on the remote GPU machine
3. Verify `curl http://127.0.0.1:8080/health` on the remote machine
4. Start Tailscale on the Jetson
5. Verify `curl http://<remote-tailscale-ip>:8080/health` from the Jetson
6. Start `phosphobot` on the Jetson
7. Open the dashboard
8. Start AI control

## 11. Example full command set

### Remote GPU machine

```bash
git clone https://github.com/phospho-app/phosphobot.git
cd phosphobot
uv sync
sudo tailscale up
tailscale ip -4
cd phosphobot
uv run --python 3.10 phosphobot serve-smolvla \
  --model-id myname/piper-smolvla \
  --host 0.0.0.0 \
  --port 8080
```

### Jetson / robot machine

```bash
git clone https://github.com/phospho-app/phosphobot.git
cd phosphobot
sudo tailscale up
curl http://100.64.0.10:8080/health
make CAN_INTERFACE=can_piper
```

Then in the dashboard set:

```txt
Inference Mode: Remote URL (Tailscale)
Remote Inference URL: http://100.64.0.10:8080
```

## 12. Troubleshooting

### `curl /health` fails from the Jetson

Check:

- both machines are connected to the same Tailscale tailnet
- the remote server is still running
- the remote server is bound to `0.0.0.0`
- port `8080` is not blocked on the remote machine

### `Start AI control` still fails

Check:

- the dashboard is set to `Remote URL (Tailscale)`
- the remote URL is exactly correct
- the selected model type is `SmolVLA`
- the model ID in the dashboard matches the model loaded by the remote server
- the remote machine logs do not show model loading or inference errors

### The remote machine has no GPU

The server may still start on CPU, but inference will be much slower.

### Private Hugging Face model cannot load

Run:

```bash
uv run hf auth login
```

on the remote machine, then restart the server.

## 13. Notes

- This setup is intended for trusted Tailscale networks.
- The remote server should stay running for as long as you want AI control available.
- The Jetson remains responsible for robot execution, camera input, and the control loop.
- The remote machine only performs model inference.
