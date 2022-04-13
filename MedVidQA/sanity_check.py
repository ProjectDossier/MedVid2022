import json
import sys
import os


def read_json(data_path):
    #### reading the json file
    with open(data_path, "r") as rfile:
        data_items = json.load(rfile)
    return data_items


def check_fields(data_path):
    #### check the json file contains the required fields
    data_items = read_json(data_path)
    for data_item in data_items:
        if "sample_id" not in data_item:
            print(
                f"submitted prediction json file doesn't have required 'sample_id' field for {data_item}!"
            )
            sys.exit(0)

        if "answer_start_second" not in data_item:
            print(
                f"submitted prediction json file doesn't have required 'answer_start_second' field for {data_item}!"
            )
            sys.exit(0)

        if "answer_end_second" not in data_item:
            print(
                f"submitted prediction json file doesn't have required 'answer_end_second' field for {data_item}!"
            )
            sys.exit(0)

        if data_item["sample_id"] == "" or data_item["sample_id"] == None:
            print("'sample_id' field can not be empty")
            sys.exit(0)

        if (
            data_item["answer_start_second"] == ""
            or data_item["answer_start_second"] == None
        ):
            print("'answer_start_second' field can not be empty")
            sys.exit(0)

        if (
            data_item["answer_end_second"] == ""
            or data_item["answer_end_second"] == None
        ):
            print("'answer_start_second' field can not be empty")
            sys.exit(0)


def check_file_name_and_type(submission_file):
    ### check the file name
    if os.path.basename(submission_file) != "predictions.json":
        print("submission json file name is invalid!")
        sys.exit(0)


def check_non_ascii_characters(submission_file_path):
    ### check for non_asciii characters
    data_item_list = read_json(submission_file_path)
    for data_item in data_item_list:
        for key, value in data_item.items():
            if not str(value).isascii():
                print(f"Non-ascii characters in the file: {value}")
                sys.exit(0)
    return True


def read_and_validate_json(submission_file_path):
    ##### to check the validatity of the submission file

    check_file_name_and_type(submission_file_path)
    check_fields(submission_file_path)
    check_non_ascii_characters(submission_file_path)

    print("Submission file validated successfully!!")


def main():
    prediction_file = sys.argv[1]
    read_and_validate_json(prediction_file)


if __name__ == "__main__":
    main()
