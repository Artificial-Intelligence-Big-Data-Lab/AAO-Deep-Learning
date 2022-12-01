NUM_GPU_1="$(sudo ps aux|grep 'esperimento_paper.py'|wc -l)";
WALK_GPU_1="$(sudo sed '1q;d' /home/unica/PhD-Market-Nets/experiments/last_walk_log_gpu_1.txt)";
ENDED_GPU_1="$(sudo sed '2q;d' /home/unica/PhD-Market-Nets/experiments/last_walk_log_gpu_1.txt)";

DATE_GPU_1="$(echo -e \"Last executed at $(date)\")";

if (($NUM_GPU_1 < 2))
then 
    if (($ENDED_GPU_1 < 1));
    then
	cd /home/unica/PhD-Market-Nets/src
	source ../venv/bin/activate
        nohup python esperimento_paper.py --start_index_walk ${WALK_GPU_1} &
        echo "${DATE_GPU_1}. It was crashed." > /home/unica/PhD-Market-Nets/src/log_gpu_1.txt
    fi;
else
    echo "${DATE_GPU_1}. It was still running." > /home/unica/PhD-Market-Nets/src/log_gpu_1.txt;
fi;


NUM_GPU_2="$(sudo ps aux|grep 'esperimento_paper_2.py'|wc -l)";
WALK_GPU_2="$(sudo sed '1q;d' /home/unica/PhD-Market-Nets/experiments/last_walk_log_gpu_2.txt)";
ENDED_GPU_2="$(sudo sed '2q;d' /home/unica/PhD-Market-Nets/experiments/last_walk_log_gpu_2.txt)";

DATE_GPU_2="$(echo -e \"Last executed at $(date)\")";

if (($NUM_GPU_2 < 2))
then 
    if (($ENDED_GPU_2 < 1));
    then
	cd /home/unica/PhD-Market-Nets/src
	source ../venv/bin/activate
        nohup python esperimento_paper_2.py --start_index_walk ${WALK_GPU_2} &
        echo "${DATE_GPU_2}. It was crashed." > /home/unica/PhD-Market-Nets/src/log_gpu_2.txt
    fi;
else
    echo "${DATE_GPU_2}. It was still running." > /home/unica/PhD-Market-Nets/src/log_gpu_2.txt;
fi;