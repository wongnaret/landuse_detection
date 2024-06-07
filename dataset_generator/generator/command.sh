python __main__.py --groupby-week --latitude-extent 16.3746943000000003 16.3128327000000013 --longitude-extent 99.6175520999999975 99.7004346999999956 --time-extent 2017-04-01 2019-12-30 --n-sample 15000 --preview -o train_set_wk
python __main__.py --groupby-week --latitude-extent 16.4834 16.4344 --longitude-extent 99.6901 99.7776 --time-extent 2017-04-01 2019-12-30 --n-sample 15000 --preview -o test_set_wk
python __main__.py --groupby-week --latitude-extent 16.5634 16.5132 --longitude-extent 99.6901 99.7457 --time-extent 2017-04-01 2019-12-30 --train-set-only --preview -o test_raster_3
python __main__.py --groupby-week --latitude-extent 16.5341 16.5017 --longitude-extent 99.7569 99.7981 --time-extent 2017-04-01 2019-12-30 --train-set-only --preview -o test_raster_4
python __main__.py --groupby-week --latitude-extent 16.53331 16.51118 --longitude-extent 99.77673 99.79688 --time-extent 2017-04-01 2019-12-30 --train-set-only --preview -o test_raster_5

python __main__.py --groupby-week --latitude-extent 16.3175 16.2907 --longitude-extent 102.1741 102.2040 --time-extent 2017-04-01 2019-12-30 --train-set-only --preview -o bk_agrimap --class-shp dataset2/shp/cliped_sugar.shp --class-shp dataset2/shp/cliped_rice.shp
python __main__.py --groupby-week --latitude-extent 16.5019 16.4301 --longitude-extent 102.0488 102.1398 --time-extent 2017-04-01 2019-12-30 --train-set-only --preview -o kc_agrimap --class-shp dataset2/shp/cliped_sugar.shp --class-shp dataset2/shp/cliped_rice.shp
python __main__.py --groupby-week --latitude-extent 14.9092 14.8559 --longitude-extent 99.7549 99.8402 --time-extent 2017-04-01 2019-12-30 --train-set-only --preview -o dc_agrimap --class-shp dataset2/shp/cliped_sugar.shp --class-shp dataset2/shp/cliped_rice.shp

python __main__.py --groupby-week --latitude-extent 16.3175 16.2907 --longitude-extent 102.1741 102.2040 --time-extent 2017-04-01 2019-12-30 --train-set-only --preview -o bk_mitrphol --class-shp dataset2/shp/bk.shp
python __main__.py --groupby-week --latitude-extent 16.5019 16.4301 --longitude-extent 102.0488 102.1398 --time-extent 2017-04-01 2019-12-30 --train-set-only --preview -o kc_mitrphol --class-shp dataset2/shp/kc.shp
python __main__.py --groupby-week --latitude-extent 14.9092 14.8559 --longitude-extent 99.7549 99.8402 --time-extent 2017-04-01 2019-12-30 --train-set-only --preview -o dc_mitrphol --class-shp dataset2/shp/dc.shp
