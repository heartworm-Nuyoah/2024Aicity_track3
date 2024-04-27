def merge_frames(input_file, output_file):

    data = {}
    with open(input_file, 'r') as f:
        for line in f:
            video_id, action_category, start_frame, end_frame = map(int, line.strip().split())
            key = (video_id, action_category)
            if key not in data:
                data[key] = []
            data[key].append((start_frame, end_frame))

    with open(output_file, 'w') as f:
    
        for key, frames_list in data.items():
            if len(frames_list) >= 2:
                print(f"Video ID: {key[0]}, Action Category: {key[1]},{frames_list}")
                frames_list.sort()  
                merged_frames = [frames_list[0]]  
                for frame in frames_list[1:]:
                    if frame[0] - merged_frames[-1][1] < 20:
                       
                        merged_frames[-1] = (merged_frames[-1][0], max(frame[1], merged_frames[-1][1]))
                    else:
                       
                        merged_frames.append(frame)
              
                for start, end in merged_frames:
                    print(f"Start Frame: {start}, End Frame: {end}")
                    f.write(f"{key[0]} {key[1]} {start} {end}\n")
            else:
                f.write(f"{key[0]} {key[1]} {frames_list[0][0]} {frames_list[0][1]}\n")


input_file = "/disk1/xyh/AICITY/2024code/mmaction2-master/submit_results_source/swin/results_submission.txt"
output_file = "/disk1/xyh/AICITY/2024code/mmaction2-master/submit_results_source/swin/results_submission_process.txt"


merge_frames(input_file, output_file)
