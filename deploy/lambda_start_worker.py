"""AWS Lambda: start the GPU server when calls are waiting in the queue.

Trigger: EventBridge Scheduler, every 15 minutes (for example cron(0/15 3-13 * * ? *) = 08:30-19:15 IST).
Environment: QUEUE_URL, INSTANCE_ID.  IAM: sqs:GetQueueAttributes on the queue, ec2:DescribeInstances,
ec2:StartInstances on the instance.  The worker on the server stops the instance itself when the queue is empty.
"""
import os

import boto3

sqs = boto3.client("sqs")
ec2 = boto3.client("ec2")


def handler(event, context):
    attrs = sqs.get_queue_attributes(QueueUrl=os.environ["QUEUE_URL"],
                                     AttributeNames=["ApproximateNumberOfMessages",
                                                     "ApproximateNumberOfMessagesNotVisible"])["Attributes"]
    waiting = int(attrs["ApproximateNumberOfMessages"]) + int(attrs["ApproximateNumberOfMessagesNotVisible"])
    inst = ec2.describe_instances(InstanceIds=[os.environ["INSTANCE_ID"]])["Reservations"][0]["Instances"][0]
    state = inst["State"]["Name"]
    if waiting > 0 and state == "stopped":
        ec2.start_instances(InstanceIds=[os.environ["INSTANCE_ID"]])
        return {"started": True, "waiting": waiting}
    return {"started": False, "waiting": waiting, "state": state}
