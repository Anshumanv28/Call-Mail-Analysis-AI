# Lambda "callai-start" (for DevOps)

Lets the call center portal switch the CallAI server on when work is waiting. About 10 minutes to set up.

1. **Create function**: Lambda → Create function → name `callai-start`, runtime **Python 3.12**, region ap-south-1.
   Paste `lambda_callai_start.py`; save it as `lambda_function.py`, handler `lambda_function.lambda_handler` (the console default).
   Configuration → General: timeout **15 s**.
2. **Environment variables**:
   - `INSTANCE_ID` = `i-0d55086ff058560c3`
   - `WAKE_TOKEN` = a long random secret (e.g. `openssl rand -hex 32`)
3. **Permissions**: add this inline policy to the function's execution role:
   ```json
   {
     "Version": "2012-10-17",
     "Statement": [
       { "Effect": "Allow", "Action": "ec2:DescribeInstances", "Resource": "*" },
       { "Effect": "Allow", "Action": "ec2:StartInstances",
         "Resource": "arn:aws:ec2:ap-south-1:<ACCOUNT_ID>:instance/i-0d55086ff058560c3" }
     ]
   }
   ```
   If the instance's EBS volume is encrypted with a customer-managed KMS key, also allow `kms:CreateGrant` on that key
   (not needed with the default `aws/ebs` key).
4. **Function URL**: Configuration → Function URL → Create → auth type **NONE** (the code checks the token).
5. **Test** (replace URL and token):
   ```
   curl -H "Authorization: Bearer <WAKE_TOKEN>" <FUNCTION_URL>                # GET  → {"state": "running"|"stopped"}
   curl -X POST -H "Authorization: Bearer <WAKE_TOKEN>" <FUNCTION_URL>        # POST → starts it if stopped
   curl <FUNCTION_URL>                                                       # no token → 401
   ```
6. Send the **Function URL** and **WAKE_TOKEN** privately to the CallAI owner; they go into the portal backend as
   `CALLAI_WAKE_URL` and `CALLAI_WAKE_TOKEN`.
