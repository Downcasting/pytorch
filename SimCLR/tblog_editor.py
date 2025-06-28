import os
import sys
from tensorboard.backend.event_processing import event_accumulator
from tensorflow.core.util import event_pb2
from tensorflow.python.lib.io import tf_record

def edit_tb_events(input_path, output_path, target_tag, new_value=None, remove=False):
    """
    TensorBoard 이벤트 파일 편집 함수.
    - target_tag: 수정/삭제할 scalar tag 이름
    - new_value: None이면 삭제, 숫자면 해당 값으로 교체
    - remove: True면 삭제, False면 교체 (new_value 필수)
    """

    # 이벤트 리더
    reader = tf_record.tf_record_iterator(input_path)
    # 이벤트 라이터
    writer = tf_record.TFRecordWriter(output_path)

    for raw_record in reader:
        event = event_pb2.Event()
        event.ParseFromString(raw_record)

        # 수정 대상 이벤트인지 체크
        if event.HasField('summary'):
            modified_summaries = []
            for val in event.summary.value:
                if val.tag == target_tag:
                    if remove:
                        # 이 태그 삭제 (skip adding)
                        print(f"Removed tag {target_tag} at step {event.step}")
                        continue
                    else:
                        # 태그 값 수정
                        if new_value is not None:
                            print(f"Modified tag {target_tag} at step {event.step} from {val.simple_value} to {new_value}")
                            val.simple_value = float(new_value)
                modified_summaries.append(val)
            # 이벤트 summary.value 교체
            del event.summary.value[:]
            event.summary.value.extend(modified_summaries)

        # 이벤트 쓰기
        writer.write(event.SerializeToString())

    writer.close()
    print(f"Edited event file saved to: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python tb_log_editor.py <input_event_file> <output_event_file> <target_tag> [new_value/remove]")
        print("Examples:")
        print("  # loss 삭제: python tb_log_editor.py events.out.tfevents.123 edited.tfevents train_loss remove")
        print("  # loss 값 변경: python tb_log_editor.py events.out.tfevents.123 edited.tfevents train_loss 0.1234")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    tag = sys.argv[3]
    remove_flag = False
    new_val = None

    if len(sys.argv) == 5:
        if sys.argv[4].lower() == 'remove':
            remove_flag = True
        else:
            try:
                new_val = float(sys.argv[4])
            except:
                print("new_value must be a float or 'remove'")
                sys.exit(1)

    edit_tb_events(input_file, output_file, tag, new_val, remove_flag)
