#!/bin/bash

# Color definitions
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if the number of arguments is correct
if [ $# -ne 1 ]; then
    echo -e "${RED}Usage: $0 <number of programs started>${NC}"
    exit 1
fi

# Read parameters
n=$1

if ! [[ "$n" =~ ^[1-9][0-9]*$ ]]; then
    echo -e "${RED}Error: parameter must be a positive integer.${NC}"
    exit 1
fi

# Record the start time
start_time=$(date '+%Y-%m-%d %H:%M:%S')
start_timestamp=$(date +%s)

# Stop all running programs
cleanup() {
    echo -e "${RED}Script stopped and is terminating all started programs...${NC}"
    kill $(pgrep -f "navigation_star_new_eval_v3-[0-9]*\.py") 2>/dev/null
    echo -e "${RED}All programs have been terminated.${NC}"
}

# Call cleanup function when receiving SIGINT, SIGTERM or EXIT
trap cleanup SIGINT SIGTERM EXIT

# Start n DGN Python programs in the background
for (( i=1; i<=n; i++ )); do
    pyfile="navigation_star_new_eval_v3-$i.py"
    echo -e "${GREEN}Starting $pyfile ...${NC}"
    nohup python3 $pyfile > navigation_star_new_eval_v3-$i.log 2>&1 &
    pids[$i]=$!
    statuses[$i]="running"
    stop_times[$i]=""
done

# Monitor the running status of the programs
echo -e "${GREEN}DGN is running...${NC}"

round=0
while true; do
    clear
    round=$((round + 1))
    current_timestamp=$(date +%s)
    elapsed_total=$((current_timestamp - start_timestamp))

    echo "============================="
    printf "Status Check Round: %-10d\n" "$round"
    printf "Elapsed Time      : %-10s\n" "$(date -ud "@$elapsed_total" +%H:%M:%S)"
    printf "Start Time        : %-10s\n" "$start_time"
    echo "============================="

    all_stopped=true
    for (( i=1; i<=n; i++ )); do
        pyfile="navigation_star_new_eval_v3-$i.py"
        if [ "${statuses[$i]}" == "running" ]; then
            if ps -p ${pids[$i]} > /dev/null; then
                cpu_usage=$(ps -p ${pids[$i]} -o %cpu=)
                mem_usage=$(ps -p ${pids[$i]} -o %mem=)
                printf "${GREEN}%s (PID: %-6d) is running.\n${NC}" "$pyfile" "${pids[$i]}"
                printf "  %-15s: %-5s%%\n" "CPU Usage" "$cpu_usage"
                printf "  %-15s: %-5s%%\n" "Memory Usage" "$mem_usage"
                all_stopped=false
            else
                stop_times[$i]=$(date '+%Y-%m-%d %H:%M:%S')
                statuses[$i]="stopped"
            fi
        fi
    done

    for (( i=1; i<=n; i++ )); do
        pyfile="navigation_star_new_eval_v3-$i.py"
        if [ "${statuses[$i]}" == "stopped" ] && [ -n "${stop_times[$i]}" ]; then
            printf "${RED}%s (PID: %-6d) stopped at %s.\n${NC}" "$pyfile" "${pids[$i]}" "${stop_times[$i]}"
        fi
    done

    if $all_stopped; then
        echo -e "${RED}All programs have stopped.${NC}"
        break
    fi

    echo "-----------------------------"
    sleep 3  # Adjust the interval as needed
done

# Display progress bar while waiting
echo -e "${GREEN}Waiting for cleanup...${NC}"
for i in {1..30}; do
    echo -n "["
    for ((j=0; j<i; j++)); do
        echo -n "#"
    done
    for ((j=i; j<3; j++)); do
        echo -n "-"
    done
    echo -n "]"
    sleep 0.1
    echo -ne "\r"
done
echo ""

echo -e "${GREEN}All programs have been stopped.${NC}"
