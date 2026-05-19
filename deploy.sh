#!/bin/bash
scp tracker.py pi@100.83.213.24:~/bed-sentry-v3/tracker.py
scp detector.py pi@100.83.213.24:~/bed-sentry-v3/detector.py
scp app.py pi@100.83.213.24:~/bed-sentry-v3/app.py
scp zone_manager.py pi@100.83.213.24:~/bed-sentry-v3/zone_manager.py
scp templates/index.html pi@100.83.213.24:~/bed-sentry-v3/templates/index.html
ssh pi@100.83.213.24 "sudo systemctl restart bed-sentry-v3"
echo "Deployed and restarted"
