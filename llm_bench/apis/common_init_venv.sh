#!/usr/bin/env bash

BRIGHT_YELLOW=$'\033[38;5;228m'
RESET=$'\033[0m'

bright_yellow() {
  "$@" 2>&1 | while IFS= read -r line || [[ -n "${line}" ]]; do
    printf '%b%s%b\n' "${BRIGHT_YELLOW}" "${line}" "${RESET}"
  done
  return "${PIPESTATUS[0]}"
}
