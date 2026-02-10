filename="data/net_watch_$(date -u +%Y%m%d%H%M).csv"
python ping_logger.py \
  --interval 1 \
  --duration-hours 0 \
  --timeout-ms 1000 \
  --add-isp-near \
  --add-global \
  --dns \
  --out "$filename"



