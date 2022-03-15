function [] = save_figure_kmj(filename, figI)
% Save figure to file
% Usage:
% save_figure('/some/folder/hello.png') % save as png
% save_figure('/some/folder/hello.ps')  % save as ps
% save_figure('/some/folder/hello')     % save as png and ps

if is_test()
    disp(['   *TESTING* [save_figure_kmj] not saving to ' filename]);
    return;
end

if ~exist('figI', 'var'); figI = gcf; end;

[filepath, name, extension] = fileparts(filename);

if ~isempty(extension)
    if ~isequal(extension, '.png') && ~isequal(extension, '.ps')
        name = [name extension];
        extension = '';
    end
end


if isempty(filepath)
    filename_no_extension = name;
else
    if ~exist(filepath, 'dir')
        mkdir(filepath);
    end
    filename_no_extension = [filepath '/' name];
end

if isempty(extension)
	extensions = {'png', 'ps'};
else
	extensions = {extension(2:end)};
end


for extension = extensions
	if strcmp(extension, 'png')
		full_filename = [filename_no_extension '.png'];
        print(figI, full_filename, '-dpng', '-painters', '-r300');
% 		saveas(figI, full_filename);
	elseif strcmp(extension, 'ps')
		full_filename = [filename_no_extension '.ps'];
		print(figI, full_filename, '-dpsc2', '-painters', '-r300');
    else
        full_filename = filename;
        saveas(figI, full_filename);
% 		error(['unrecognized extension: ' extension]);
	end
	disp(['[save_figure_kmj] figure saved as ' full_filename]);
end

end
