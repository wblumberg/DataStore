Construct the variables database:
python inventory_variables_db.py --input-root ~/data/base/model/ --exclude-dirs ~/data/base/model/ecmwf_hr/ --output-variables variables.json

python build_member_zarr.py --input-root ~/data/base/model/wrf4nssl/ --member wrf4nssl --output-zarr wrf4nssl --variables-db variables.json --max-lags 2 --max-times 3
python build_member_zarr.py --input-root ~/data/base/model/hrrr/ --member hrrr --output-zarr ./href_members --variables-db variables.json --max-lags 2 --max-times 3