#!/bin/sh
# nb - the agent's dated notebook. agent.sh's TOOL recall already
# searches /root/nb, but until now nothing could write there.
#
# Usage:
#   nb add <words...>      append a datestamped line to today's note
#   nb list [n]            show the last n lines (default 10)
#   nb search <words...>   case-insensitive grep over all notes
#   nb days                list which days have notes
NB=/root/nb
mkdir -p "$NB"
cmd=$1; shift 2>/dev/null || true
case "$cmd" in
  add)
    [ -z "$*" ] && { echo 'usage: nb add <words...>'; exit 1; }
    echo "$(date '+%F %T') | $*" >> "$NB/$(date +%F).md"
    echo "noted" ;;
  list)
    n=$1; case "$n" in ''|*[!0-9]*) n=10 ;; esac
    cat "$NB"/*.md 2>/dev/null | tail -n "$n" ;;
  search)
    [ -z "$*" ] && { echo 'usage: nb search <words...>'; exit 1; }
    pat=$(printf '%s' "$*" | sed 's/ /|/g')
    grep -iE "$pat" "$NB"/*.md 2>/dev/null || echo "(no notes match: $*)" ;;
  days)
    ls "$NB" 2>/dev/null || echo "(no notes yet)" ;;
  *)
    echo 'usage: nb add <words...> | nb list [n] | nb search <words...> | nb days'
    exit 1 ;;
esac
