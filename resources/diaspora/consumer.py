import json
import os
import time
import csv
from diaspora_stream.api import Driver
print("about to start", flush=True)

driver_options = {
    "root_path": "/tmp/diaspora-data/",
}
driver = Driver(backend="files", options=driver_options)
# create a topic
topic_name = "interception"
consumer_name = "flowcept"
topic = driver.open_topic(topic_name)

consumer = topic.consumer(name=consumer_name)

pulls = []
events = []
count = 0


# Get the list of files in the current directory
csv_files = [file for file in os.listdir() if file.endswith('.csv')]
threshold = len(csv_files)
print("about to start with breakpoint ",threshold, flush=True)
while True:
    data = []
    metadata = []
    t1 = time.time()
    f = consumer.pull()
    print("this is f", f, type(f))
    event = f.wait(timeout_ms=1)
    t2 = time.time()
    e = event.metadata
    print("this is the event", e)


    events.append(e)
    pulls.append(t2 - t1)
    # print("h: ", e.keys(),flush=True)

    # break

    if "type" in e.keys() and 'info' in e.keys():
        if e['type'] == 'flowcept_control':
            if e['info'] == "mq_dao_thread_stopped":
                print(e,flush=True)
                count += 1
                with open("data.json", 'w') as f:
                    json.dump(events, f, indent=4)
                with open("pulls.csv", "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(pulls)

    if count == threshold:
        break





