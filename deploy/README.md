# Queue mode (server runs only while calls are waiting)

1. **SQS queue** `callai-calls` (standard), visibility timeout 30 min, with a dead-letter queue (`maxReceiveCount` 5).
   The portal sends one message per answered call: `{"call_id": "123", "recording_url": "https://.../x.wav"}`.
2. **Results**: the worker POSTs `{"results": [...]}` (portal column names + `call_id`, `confidence`,
   `needs_review`, `processed_at`) to `OTR_RESULTS_URL`, and/or writes one JSON file per batch to `OTR_RESULTS_S3`.
   A message is deleted only after its results were delivered; failures go back to the queue.
3. **Server**: install as in the main README, then:
   - copy `worker.env.example` to `/etc/callai/worker.env` and fill it in;
   - `sudo cp deploy/callai-worker.service /etc/systemd/system/ && sudo systemctl enable --now callai-worker`;
   - allow the service user to stop the machine: `echo 'callai ALL=(root) NOPASSWD: /sbin/shutdown' | sudo tee /etc/sudoers.d/callai`;
   - set the instance's shutdown behaviour to **Stop** (the default for EBS-backed instances).
   The worker stops the server after `OTR_IDLE_STOP_S` (180) seconds with an empty queue.
4. **Starter**: Lambda `lambda_start_worker.py` (env `QUEUE_URL`, `INSTANCE_ID`) on an EventBridge schedule every
   15 minutes, e.g. `cron(0/15 3-13 * * ? *)` = 08:30-19:15 IST. It starts the instance when messages are waiting.
5. **Instance role** additions: `sqs:ReceiveMessage`, `sqs:DeleteMessage`, `sqs:ChangeMessageVisibility` on the queue;
   `s3:PutObject` on the results prefix if `OTR_RESULTS_S3` is used.

Test without AWS: `python worker.py --queue-dir test_q --results-dir test_out --fake-pipeline --once`
(put `{"call_id": "1", "recording_url": "<path or URL>"}` files in `test_q/`).

# API mode with auto start / stop (what the call center portal uses)

The portal calls the API (`/process` for calls, `/llm` for mail) and starts the machine itself when a backlog builds
up (its `callaiDispatcher` calls a "callai-start" Lambda). On the server:

1. `sudo mkdir -p /etc/callai && sudo cp deploy/api.env.example /etc/callai/api.env` and fill it in
   (`CALLAI_TOKEN` = a long random secret, the same value goes into the portal's `CALLAI_TOKEN`;
   `OTR_IDLE_STOP_S=300` switches the machine off after 5 idle minutes, 0 = never). `sudo chmod 600 /etc/callai/api.env`.
2. `echo 'ubuntu ALL=(root) NOPASSWD: /sbin/shutdown' | sudo tee /etc/sudoers.d/callai-shutdown`
3. `sudo cp deploy/callai-api.service /etc/systemd/system/ && sudo systemctl enable --now callai-api`
4. Instance shutdown behaviour must be **Stop**. Security group: TCP 8700 only from the portal backend.
