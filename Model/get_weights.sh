#!/bin/bash

# Download the models weights
wget "https://downloader.disk.yandex.ru/disk/ba2e39f1266f54c1ddbe56be856577cfbef0a2a855b16ac2c1870ba956b82008/68960e86/SaV1a6hFL4ybFM_rvNx7CTAIO3D_Gt1gtmIFJYQGeB9yYeDK59ZC-MlITpJqq15RlUMjBYsbKqD4lFJ8yokoBA%3D%3D?uid=0&filename=CascadeCardioUnetWeights.zip&disposition=attachment&hash=S4iPC3bA1TiAVogpDY/aNUo/MJVZwYlKlmT/zs6elSJNkNvt7o2DSOV5TMRPWbAKq/J6bpmRyOJonT3VoXnDag%3D%3D%3A&limit=0&content_type=application%2Fzip&owner_uid=45998248&fsize=58462343&hid=f26c578488e749892e8b9150b524079c&media_type=compressed&tknv=v3" -O CardioCascadeUnetWeights.zip
unzip CascadeCardioUnetWeights.zip
rm CascadeCardioUnetWeights.zip