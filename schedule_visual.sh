MEGATRON_PATH=${1}
TENSORBOARD_DIR=${2}
ITERATION=${3}
SCHEDULE_VISUAL=true
while [ ${SCHEDULE_VISUAL} = true ]
do
    ITERATION=$((${ITERATION}+1))
    prefix="iteration ${ITERATION}"
    if [ -f "${TENSORBOARD_DIR}/Timecond" ]; then
        if grep -q "${prefix}" "${TENSORBOARD_DIR}/Timecond" ; then
            python ${MEGATRON_PATH}/schedule_visual.py \
                --input ${TENSORBOARD_DIR}/Timecond \
                --output ${TENSORBOARD_DIR}/ScheduleVisual.jpg \
                --iteration ${ITERATION}
            echo "Schedule visualization finished !!!"
            SCHEDULE_VISUAL=false
        fi
    fi
    if [[ ${SCHEDULE_VISUAL} = true ]]; then
        sleep 1m
    fi
done