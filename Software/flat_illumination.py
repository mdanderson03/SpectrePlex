import win32com.client

scheduler = win32com.client.Dispatch('Schedule.Service')
scheduler.Connect()


def run_task(task_name):
    task_folder = scheduler.GetFolder('\\')
    try:
        task = task_folder.GetTask(task_name)
        if task.Enabled:
            task_run = task.Run('')
        else:
            print("Task is disabled and cannot be run.")
    except Exception as e:
        print("Error running task:", e)

run_task('rasp_wifi_connect')