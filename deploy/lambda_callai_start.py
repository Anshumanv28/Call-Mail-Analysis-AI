"""AWS Lambda "callai-start": switch the CallAI GPU server on when the call center portal asks.

The portal's dispatcher (backend/src/workers/callaiDispatcher.js) POSTs here, with "Authorization: Bearer <token>",
when calls and mails pile up while CallAI is off. CallAI switches itself off again when idle.

Environment variables:
  INSTANCE_ID   the CallAI EC2 instance, e.g. i-0d55086ff058560c3
  WAKE_TOKEN    long random secret; the portal sends the same value as CALLAI_WAKE_TOKEN
Runtime: Python 3.12. Timeout: 15 s. Exposed through a Lambda Function URL (auth type NONE; the token is checked here).
IAM: ec2:DescribeInstances (any), ec2:StartInstances on this instance only (see LAMBDA.md).
"""
import hmac
import json
import os

import boto3

ec2 = boto3.client("ec2")


def _reply(code, body):
    return {"statusCode": code, "headers": {"Content-Type": "application/json"}, "body": json.dumps(body)}


def lambda_handler(event, context):
    headers = {k.lower(): v for k, v in (event.get("headers") or {}).items()}
    expected = "Bearer " + os.environ["WAKE_TOKEN"]
    if not hmac.compare_digest(headers.get("authorization", ""), expected):
        return _reply(401, {"error": "unauthorised"})

    iid = os.environ["INSTANCE_ID"]
    state = ec2.describe_instances(InstanceIds=[iid])["Reservations"][0]["Instances"][0]["State"]["Name"]
    method = (event.get("requestContext", {}).get("http", {}) or {}).get("method", "POST")
    if method == "GET":                      # status only
        return _reply(200, {"instance": iid, "state": state})
    if state == "stopped":
        ec2.start_instances(InstanceIds=[iid])
        return _reply(200, {"instance": iid, "state": "pending", "started": True})
    # running / pending: nothing to do; stopping: the portal will ask again after its cooldown
    return _reply(200, {"instance": iid, "state": state, "started": False})


handler = lambda_handler   # alias, for a handler setting of "<file>.handler"
