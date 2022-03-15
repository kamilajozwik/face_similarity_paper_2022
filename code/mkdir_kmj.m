function [ ] = mkdir_kmj( directory )
% same as matlab's mkdir(), but checks if it exists before creating (so we don't get a warning)

if ~isdir(directory)
    if is_test()
        disp(['   *TESTING* [mkdir_kmj] NOT creating dir ' directory]);
    else
        directory
        mkdir(directory);
        disp(['[mkdir_kmj] created dir ' directory]);
    end
end

end
