__kernel void add_numbers(__global float4* data, __global float* group_result) {

   __local float* local_result;
      
   float sum;
   float4 input1, input2, sum_vector;
   uint global_addr, local_addr;

   global_addr = get_global_id(0) * 2;
   input1 = data[global_addr];
   input2 = data[global_addr+1];
   sum_vector = input1 + input2;

   local_addr = get_local_id(0);
   local_result[local_addr] = sum_vector.s0 + sum_vector.s1 + 
                              sum_vector.s2 + sum_vector.s3; 
   barrier(CLK_LOCAL_MEM_FENCE);

   if(get_local_id(0) == 0) {
      sum = 0.0f;
      for(int i=0; i<get_local_size(0); i++) {
         sum += local_result[i];
      }
      group_result[get_group_id(0)] = sum;
   }
}

__constant sampler_t sampler_const =
        CLK_NORMALIZED_COORDS_FALSE |
        CLK_ADDRESS_NONE |
        CLK_FILTER_NEAREST;

__kernel void blue(read_only image2d_t image, write_only image2d_t result) {
        int2 coord = (int2) (get_global_id(0),get_global_id(1));

        float4 pixel = read_imagef(image,sampler_const,coord);

        pixel += (float4) (0.0, 0.0, 0.1, 0.0);

        write_imagef(result,coord,pixel);
}

__kernel void convolute(read_only image2d_t src, write_only image2d_t result, __global float4 * flt) {
        int fIndex = 0;
        float4 sum = (float4) 0.0;
        int2 coord = (int2) (get_global_id(0),get_global_id(1));


        for (int i = -KERNEL_SIZE_HALF; i <= KERNEL_SIZE_HALF; i++) {
                for(int j = -KERNEL_SIZE_HALF; j <= KERNEL_SIZE_HALF; j++) {
                        int2 local_coord = (int2)(coord.x + i,coord.y + j);
                        float4 pixel = read_imagef(src,sampler_const,local_coord);
                        sum += pixel * flt[fIndex];
                        fIndex++;
                } 
        }

        write_imagef(result,coord,sum);
}


__kernel void convolute_mem(__read_only image2d_t src, __write_only image2d_t result, __global float4 * flt) {
        int2 global_coord = (int2) (get_global_id(0),get_global_id(1));
        int2 local_coord = (int2) (get_local_id(0),get_local_id(1));
        int2 local_size = (int2) (get_local_size(0),get_local_size(1));
        int local_array = local_coord.x * ROW_SIZE + local_coord.y;
        
        // Need to think of the size of the array
        __local float4 pixels[KERNEL_SIZE_HALF * KERNEL_SIZE_HALF * 4 + ROW_SIZE * KERNEL_SIZE_HALF * 2 + COL_SIZE * KERNEL_SIZE_HALF * 2 + COL_SIZE * ROW_SIZE];

        pixels[local_array] = read_imagef(src,sampler_const,local_coord);

        if (
		get_global_id(0) < KERNEL_SIZE_HALF 			|| 
		get_global_id(0) > COL_SIZE - KERNEL_SIZE_HALF - 1		|| 
		get_global_id(1) < KERNEL_SIZE_HALF			||
		get_global_id(1) > ROW_SIZE - KERNEL_SIZE_HALF - 1
	)
	{
		// no computation for me, sync and exit
		barrier(CLK_LOCAL_MEM_FENCE);
		return;
	}

        // Places to load the image to local memory
        if(local_coord.x == 0) { // Left side of the image

                if(local_coord.y == 0) { // Case where the position is near the top left corner of the image
                        for(int i = - KERNEL_SIZE_HALF; i <= 0; i++) {
                                for(int j = - KERNEL_SIZE_HALF; j <= 0; j++) {
                                        int2 local_mem = (int2) (local_coord.x + j, local_coord.y + i);
                                        int2 global_mem = (int2) (global_coord.x + j, global_coord.y + i);
                                        pixels[local_mem.y * ROW_SIZE + local_mem.x] = read_imagef(src,sampler_const,global_mem);
                                }
                        }

                        // for(int i = 0; i <= KERNEL_SIZE_HALF; i++) {
                        //         for(int j = - KERNEL_SIZE_HALF; j <= 0; j++) {
                        //                 int2 local_mem = (int2) (local_coord.x + j, local_coord.y + i);
                        //                 int2 global_mem = (int2) (global_coord.x + j, global_coord.y + i);
                        //                 pixels[local_mem.y * ROW_SIZE + local_mem.x] = read_imagef(src,sampler_const,global_mem);
                        //         }
                        // }
                }
                else if (local_coord.y == COL_SIZE) { // Case where the position is near the bottom left corner of the image
                        for(int i = 0 ; i <= KERNEL_SIZE_HALF; i++) {
                                for(int j = - KERNEL_SIZE_HALF; j <= 0; j++) {
                                        int2 local_mem = (int2) (local_coord.x + j, local_coord.y + i);
                                        int2 global_mem = (int2) (global_coord.x + j, global_coord.y + i);
                                        pixels[local_mem.y * ROW_SIZE + local_mem.x] = read_imagef(src,sampler_const,global_mem);
                                }
                        }

                        // for(int i = - KERNEL_SIZE_HALF; i <= 0; i++) {
                        //         for(int j = - KERNEL_SIZE_HALF; j < 0; j++) {
                        //                 int2 local_mem = (int2) (local_coord.x + j, local_coord.y + i);
                        //                 int2 global_mem = (int2) (global_coord.x + j, global_coord.y + i);
                        //                 pixels[local_mem.y * ROW_SIZE + local_mem.x] = read_imagef(src,sampler_const,global_mem);
                        //         }
                        // }
                }
                else { // When X is close to the left, but far from the bottom or top of the image
                        for(int j = - KERNEL_SIZE_HALF; j < 0; j++) {
                                int2 local_mem = (int2) (local_coord.x + j, local_coord.y);
                                int2 global_mem = (int2) (global_coord.x + j, global_coord.y);
                                pixels[local_mem.y * ROW_SIZE + local_mem.x] = read_imagef(src,sampler_const,global_mem);
                        }
                }

        }
        else if (local_coord.x == local_size.x-1) { // Right side of the image

                if(local_coord.y == 0) {
                        // Case where the position is near the top right corner of the image
                        for(int i = - KERNEL_SIZE_HALF; i <= 0; i++) {
                                for(int j = 0; j <= KERNEL_SIZE_HALF; j++) {
                                        int2 local_mem = (int2) (local_coord.x + j, local_coord.y + i);
                                        int2 global_mem = (int2) (global_coord.x + j, global_coord.y + i);
                                        pixels[local_mem.y * ROW_SIZE + local_mem.x] = read_imagef(src,sampler_const,global_mem);
                                }
                        }

                        // for(int i = 0; i <= KERNEL_SIZE_HALF; i++) {
                        //         for(int j = 0; j <= KERNEL_SIZE_HALF; j++) {
                        //                 int2 local_mem = (int2) (local_coord.x + j, local_coord.y + i);
                        //                 int2 global_mem = (int2) (global_coord.x + j, global_coord.y + i);
                        //                 pixels[local_mem.y * ROW_SIZE + local_mem.x] = read_imagef(src,sampler_const,global_mem);
                        //         }
                        // }
                }
                else if (local_coord.y + KERNEL_SIZE_HALF > COL_SIZE) {
                        // Case where the position is near the bottom right corner of the image
                        for(int i = 0 ; i <= KERNEL_SIZE_HALF; i++) {
                                for(int j = 0; j <= KERNEL_SIZE_HALF; j++) {
                                        int2 local_mem = (int2) (local_coord.x + j, local_coord.y + i);
                                        int2 global_mem = (int2) (global_coord.x + j, global_coord.y + i);
                                        pixels[local_mem.y * ROW_SIZE + local_mem.x] = read_imagef(src,sampler_const,global_mem);
                                }
                        }

                        // for(int i = - KERNEL_SIZE_HALF; i <= 0; i++) {
                        //         for(int j = 0; j < KERNEL_SIZE_HALF; j++) {
                        //                 int2 local_mem = (int2) (local_coord.x + j, local_coord.y + i);
                        //                 int2 global_mem = (int2) (global_coord.x + j, global_coord.y + i);
                        //                 pixels[local_mem.y * ROW_SIZE + local_mem.x] = read_imagef(src,sampler_const,global_mem);
                        //         }
                        // }
                }
                else {
                        // When X is close to the right, but far from the bottom or top of the image
                        for(int j = 1; j < KERNEL_SIZE_HALF; j++) {
                                int2 local_mem = (int2) (local_coord.x + j, local_coord.y);
                                int2 global_mem = (int2) (global_coord.x + j, global_coord.y);
                                pixels[local_mem.y * ROW_SIZE + local_mem.x] = read_imagef(src,sampler_const,global_mem);
                        }
                }
        }
        else if(local_coord.y == 0) {
                // Case where the position is near the top, but far from the left or right corner of the image
                for(int i = - KERNEL_SIZE_HALF; i < 0; i++){
                        int2 local_mem = (int2) (local_coord.x, local_coord.y + i);
                        int2 global_mem = (int2) (global_coord.x, global_coord.y + i);
                        pixels[local_mem.y * ROW_SIZE + local_mem.x] = read_imagef(src,sampler_const,global_mem); 
                }
        }
        else if(local_coord == local_size.y - 1) {
                // Case where the position is near the bottom, but far from the left or right corner of the image
                for(int i = 1; i < KERNEL_SIZE_HALF; i++){
                        int2 local_mem = (int2) (local_coord.x, local_coord.y + i);
                        int2 global_mem = (int2) (global_coord.x, global_coord.y + i);
                        pixels[local_mem.y * ROW_SIZE + local_mem.x] = read_imagef(src,sampler_const,global_mem); 
                }
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);

        //Calculate Convolution
        int fIndex = 0;
        float4 sum = (float4) 0.0;
        for (int i = -KERNEL_SIZE_HALF; i <= KERNEL_SIZE_HALF; i++) {
                for(int j = -KERNEL_SIZE_HALF; j <= KERNEL_SIZE_HALF; j++) {
                        int2 local_mem = (int2)(local_coord.x + i,local_coord.y + j);
                        float4 pixel = pixels[local_mem.y * ROW_SIZE + local_mem.x];
                        sum += pixel * flt[fIndex];
                        fIndex++;
                } 
        }

        write_imagef(result,global_coord,sum);

}