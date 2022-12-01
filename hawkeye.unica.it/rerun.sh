PROCESS="$(sudo ps aux|grep 'web-api.py'|wc -l)";

if (($PROCESS < 2))
then 
    cd /home/ubuntu/PhD-Market-Nets/src
    nohup python3 web-api.py &
    echo "Script launched" > /home/ubuntu/cronlog.txt;
else
    echo "It is still running." > /home/ubuntu/cronlog.txt;
fi;
