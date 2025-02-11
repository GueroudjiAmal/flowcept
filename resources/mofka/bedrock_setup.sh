export HG_LOG_LEVEL=error
export FI_LOG_LEVEL=Trace
rm -rf mofka.json
bedrock cxi -c resources/mofka/mofka_config.json &
sleep 0.3
mofkactl topic create interception --groupfile mofka.json
mofkactl partition add interception --type memory --rank 0 --groupfile mofka.json
sleep 0.3
echo "Created topic."
while true; do sleep 3600; done