To upload files after they have been generated use:

```
aws s3 sync confirmed_products/ s3://dist-s1-calval/amy-pickenson-10-sites-processed-on-september-2025/confirmed_products --exclude ".DS_Store" --exclude "*/.DS_Store" --profile <PROFILE>
```
and
```
aws s3 sync dist_hls/ s3://dist-s1-calval/amy-pickenson-10-sites-processed-on-september-2025/dist_hls --exclude ".DS_Store" --exclude "*/.DS_Store" --profile <PROFILE>
```

The `confirmed_proudcts` are `dist-s1` products and there are currently 10 directories labeled according to `<DISTURBANCE CATEGORY>__<MGRS_TILE_ID>`.
Similarly, `dist_hls` are the `dist-hls` products.