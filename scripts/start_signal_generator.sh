#!/bin/bash
cd /home/avinash-kumar/tradding_app
./venv/bin/python scripts/live_paper_trading.py >> /tmp/signal_generator.log 2>&1
