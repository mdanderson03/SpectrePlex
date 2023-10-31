import PySimpleGUI as sg
import time
import os 
from GUI_layout import *
from autocyplex import *


window = sg.Window('AutoCyPlex', layout)
layout = 1 # Default visible layout
list_status = []
message_dict = {1:"a Full Cycle"}

microscope = cycif()
pump = fluidics(6, 3)
offset_array = [0, -8, -7, -11.5]
z_slices = 7


while True:
    event, values = window.read()
    
    if event == None:
        break
    elif event == "Start Run":

        # ----- Full Cycle ----- #

        message = "-message-"
        experiment_directory = values["-path-"]
        start_cycle = int(values["-cycle-"])
        end_cycle = int(values["-cycle_end-"])
        incub_val = 0

        if layout == 1:
            if values["-incub_time-"] == True:
                try: 
                    incub_val = int(values["-fc_incub_input-"])
                    if incub_val == 0:
                        sg.PopupError("Invalid incubation time")
                        continue
                    else:
                        pass
                except:
                    if values["-fc_incub_input-"] == "":
                        sg.PopupError("Missing Inputs")
                        continue
                    else: 
                        sg.PopupError("Invalid incubation time")
                        continue

            if values["-checkbox-"] == True:
                try: 
                    os.listdir(experiment_directory)
                    if start_cycle >= end_cycle:
                        sg.PopupError("End cycle must be larger than start cycle") 
                    else: 
                        if sg.popup_yes_no(f'Are you sure you want to start {message_dict[layout]}?') != "Yes":
                            sg.popup("Run Canceled")
                        else:
                            # --- multi-cycle run --- #
                            window["-frame_list-"].update(value="Status: currently running")
                            window.refresh()
                            window["-PBAR-"].update(visible=True, current_count=0)
                            bar_value = 10/((end_cycle-start_cycle)+1)
                            i = 0
                            for cycle in range(start_cycle,end_cycle+1):
                                if incub_val != 0:
                                    microscope.full_cycle(experiment_directory, cycle, offset_array, cycle, pump, z_slices, window, list_status, incub_val)
                                else:
                                    microscope.full_cycle(experiment_directory, cycle, offset_array, cycle, pump, z_slices, window, list_status)
                                    
                                
                                i = i + bar_value
                                window["-PBAR-"].update(current_count=i)
                                window.refresh()
                            
                            window["-PBAR-"].update(visible=False)
                            window["-frame_list-"].update(value="Status: idle")
                            status_update("Done!", list_status, window)
                            
                except:
                    sg.PopupError("Missing Inputs")

            else: 
                try: 
                    os.listdir(experiment_directory)
                    if sg.popup_yes_no(f'Are you sure you want to start {message_dict[layout]}?') != "Yes":
                        sg.popup("Run Canceled")
                    else:
                        # --- one-cycle run --- #
                        window["-frame_list-"].update(value="Status: currently running")
                        window.refresh()
                        for cycle in range(start_cycle,start_cycle+1):   
                            if incub_val != 0:
                                microscope.full_cycle(experiment_directory, cycle, offset_array, cycle, pump, z_slices, window, list_status, incub_val)
                            else:
                                microscope.full_cycle(experiment_directory, cycle, offset_array, cycle, pump, z_slices, window, list_status)
                        
                        window["-frame_list-"].update(value="Status: idle")
                        status_update("Done!", list_status, window)

                except:
                    sg.PopupError("Missing Inputs")
 

    # ----------- PBS On button function ----------- # 
    elif event == "-pbs_on-":
        mess_str = "PBS on"
        pump.liquid_action("PBS_flow_on")
        window["-frame_list-"].update(value="Status: currently running")
        status_update(mess_str, list_status, window)
        window.refresh()


    # ----------- PBS off button function ----------- #
    elif event == "-pbs_off-":
        mess_str = "PBS off"
        window["-frame_list-"].update(value="Status: currently running")
        status_update(mess_str, list_status, window)
        window.refresh()
        pump.liquid_action("PBS_flow_off")
        time.sleep(2)
        window["-frame_list-"].update(value="Status: idle")
        window.refresh()
    
    # ----------- full cycle incubation checkbox function ----------- #
    elif event == "-incub_time-":
        if values["-incub_time-"] == True:
            window["-fc_incub_txt-"].update(visible=True)
            window["-fc_incub_input-"].update(visible=True)
        else: 
            window["-fc_incub_txt-"].update(visible=False)
            window["-fc_incub_input-"].update(visible=False)
    
    # ----------- updates available End Cycle Number options based on Start Cycle Number value ----------- # 
    elif event == "-cycle-":
        new_list = []
        cycle_options = ["1","2","3","4","5","6","7","8","9","10"]
        for i in cycle_options:
            if int(i) > int(values["-cycle-"]):
                new_list.append(i)
            else:
                pass
        cycle_options=new_list
        try:
            if int(values["-cycle-"]) >= int(values["-cycle_end-"]):
                window["-cycle_end-"].update(values=cycle_options, value=cycle_options[0])
            else:
                window["-cycle_end-"].update(values=cycle_options, value=values["-cycle_end-"]) 
        except:
            window["-cycle_end-"].update(values=cycle_options)
            window.refresh()

 # ----------- multi-round checkbox function ----------- #
    elif event == "-checkbox-":
        if values["-checkbox-"] == True:
            window["-txt_end-"].update(visible=True)
            window["-cycle_end-"].update(visible=True)
            cycle_options = ["1","2","3","4","5","6","7","8","9","10"]
            num_list = ["1","2","3","4","5","6","7","8","9"]
            window["-cycle_end-"].update(value="8", values=cycle_options)
            window["-cycle-"].update(value="0", values=num_list)
        else:
            cycle_options = ["1","2","3","4","5","6","7","8","9","10"]
            window["-cycle-"].update(value="0",values=cycle_options)
            window["-txt_end-"].update(visible=False)
            window["-cycle_end-"].update(visible=False)

# ----------- makes new folder in specified parent directory ----------- #
    elif event == "-new_folder-":
        try:
            parent_dir = values["-path-"]
            directory = values["-new_path-"]
            if parent_dir == "":
                sg.PopupError("No parent directory specified")
            else:
                path = os.path.join(parent_dir, directory)
                os.mkdir(path)
                window["-path-"].update(path)
                window["-new_path-"].update("")
        except:
            sg.PopupError("Unable to make folder.\nFolder might already exist", )

window.close()