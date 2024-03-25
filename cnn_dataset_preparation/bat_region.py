#!/usr/bin/env python
import sys
import os
import ice_thickness as thickness
import basal_melting as melting
import velocity as vel
import sea_ice as sea_ice

def process_region(region_number):
    thickness(region_number)
    melting(region_number)
    vel(region_number)
    sea_ice(region_number)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: bat_region.py <region_number>")
        sys.exit(1)
    
    region_number = int(sys.argv[1])
    process_region(region_number)
