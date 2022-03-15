function [ ] = save_mat_kmj( varargin )
% same as matlab's save(), but also creates directories if needed.
% example:
%   save_mat_kmj('test.mat', 'RDM', 'x');

filename = varargin{1};

if is_test()
    disp(['   *TESTING* [save_mat_kmj] not saving to ' filename]);
    return;
end

variable_names = {};
for i = 2:length(varargin)
    variable_name = varargin{i};
    variable_value = evalin('caller', variable_name);
    variable_names{end+1} = variable_name;
    eval([variable_name ' = variable_value;']); % evil code to create local variable to make save() work
end

[filedir, ~, ~] = fileparts(filename);
mkdir_kmj(filedir);

save(varargin{:});

if isempty(strfind(filename, '.mat'))
    filename = [filename '.mat'];
end
disp(['[save_mat_kmj] saved ' strjoin(variable_names, ',') ' in ' filename]);

end

