export model=$1
export attack=$2
export offset=$3
export dataset=$4
export folder=../bash_scripts/out_${model}_${attack}_direct # 打印文件

cd ../testcase
# 创建测试数据json
if [ ! -f ${model}_${attack}_direct.json ]; then
    cp blank_direct_100.json ${model}_${attack}_direct.json
    echo "File ${model}_${attack}_direct.json created."
else
    echo "File ${model}_${attack}_direct.json already exists."
fi

cd ../launch_scripts

if [ ! -d $folder ]; then
    mkdir $folder
    echo "Folder $folder created."
else
    echo "Folder $folder already exists."
fi
time=$(date +"%Y%m%d_%H%M%S")
bash run_individual.sh $model $attack direct 8 $dataset $offset > ${folder}/${offset}_${time}.out &
# nohup bash run_individual.sh $model $attack direct 8 $offset > ${folder}/${offset}.out &

wait
