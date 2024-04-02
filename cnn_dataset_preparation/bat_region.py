#!/usr/bin/env python
import sys
import os
import thickness as thickness
import basal_melting as melting
import velocity as vel
import sea_ice_concentration as sea_ice

print(f'Processing region {sys.argv[1]}, we are into bat_region.py')

def process_region(region_number):
    #thickness.thickness(region_number)
    #melting.basal_melting(region_number)
    vel.velocity(region_number)
    sea_ice.sea_ice_concentration(region_number)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: bat_region.py <region_number>")
        sys.exit(1)

    region_number = int(sys.argv[1])
    process_region(region_number)
    