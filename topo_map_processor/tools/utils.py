
import resource


def relax_open_file_limit():

    # Get the current soft and hard limits for open files
    current_soft, current_hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    print(f"Current open file limits: Soft={current_soft}, Hard={current_hard}")
    
    # Set new soft and hard limits for open files
    new_soft = resource.RLIM_INFINITY  # Set to unlimited
    new_hard = resource.RLIM_INFINITY  # Set to unlimited

    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, new_hard))
        print(f"New open file limits set: Soft={new_soft}, Hard={new_hard}")
    except Exception as e:
        print(f"Error setting resource limit to unlimited: {e}")
        print('Increasing soft limit to match the hard limit instead')
        new_soft = current_hard
        new_hard = current_hard
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, new_hard))
            print(f"New open file limits set: Soft={new_soft}, Hard={new_hard}")
        except Exception as e:
            print(f"Error setting soft resource limit to match hard limit: {e}")
    
    # Verify the new limits
    updated_soft, updated_hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    print(f"Verified open file limits: Soft={updated_soft}, Hard={updated_hard}")
