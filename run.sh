cat *.pid | xargs kill -9
rm *.pid
python3 test.py & echo $! > $!.pid
