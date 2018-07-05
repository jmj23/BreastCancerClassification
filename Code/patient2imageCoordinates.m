%%  function image_coordinates = patient2imageCoordinates(patient_coordinates, header)
%   This function converts dicom coordinates from the patient reference
%   frame to the image reference frame.
%
%   Inputs:
%       patient_coordinates: the [x; y; z] location in the patient reference system 
%       header: the header (obtained using dicomread) of the 1st image in
%               the image stack that you want the patient coordinates to match. 
%
%   Ouputs
%       image_coordinates:  the [x; y; z] location in the image reference system
%
function image_coordinates = patient2imageCoordinates(patient_coordinates, header)

    %identify GE vs. DICOM header
    if isfield(header, 'RawHeader') %
        if (header.SeriesData.position==2 && header.SeriesData.entry ==2)
            warning('Have not verified Image Orientation when read from Pfiles');
            
            %Get the basis vectors for the image coordinate space:
%             x_basis = [-1; 0; 0];
%             y_basis = [0; -1; 0];
%             z_basis = cross(x_basis, y_basis);

              x_basis = ([header.ImageData.trhc_R; header.ImageData.trhc_A; header.ImageData.trhc_S] - ...
                        [header.ImageData.tlhc_R; header.ImageData.tlhc_A; header.ImageData.tlhc_S]) .* [-1; 1; -1]*1/header.ImageData.dfov;
                    
              y_basis = ([header.ImageData.trhc_R; header.ImageData.trhc_A; header.ImageData.trhc_S] - ...
                        [header.ImageData.brhc_R; header.ImageData.brhc_A; header.ImageData.brhc_S]) .* [-1; 1; -1]*1/header.ImageData.dfov;

              z_basis = [header.ImageData.norm_R; header.ImageData.norm_A; header.ImageData.norm_S];
        end
        
        delta_i = header.ImageData.pixsize_X;
        delta_j = header.ImageData.pixsize_Y;
        delta_k = header.ImageData.slthick - header.ImageData.scanspacing;
        
        image_origin = [header.ImageData.normal_L; header.ImageData.normal_P; header.ImageData.normal_S];
        
    else

        %Get the basis vectors for the image coordinate space:
        x_basis = [header.ImageOrientationPatient(1); header.ImageOrientationPatient(2); header.ImageOrientationPatient(3)];
        y_basis = [header.ImageOrientationPatient(4); header.ImageOrientationPatient(5); header.ImageOrientationPatient(6)];
        z_basis = cross(x_basis, y_basis);

        delta_i = header.PixelSpacing(1);
        delta_j = header.PixelSpacing(2);
        delta_k = header.SpacingBetweenSlices;

        image_origin = header.ImagePositionPatient;
    
    end

    %Transformation matrix image coordinates to patient coordinates:
    tMatrix = [x_basis*delta_i, y_basis*delta_j, z_basis*delta_k, image_origin; ...
               0,               0,               0,               1]; 


    image_coordinates = tMatrix\[patient_coordinates; 1]; 
    image_coordinates = image_coordinates(1:3);


end




