#!/usr/bin/bash
lspci -k | grep -A 2 -E "(VGA|3D)"
