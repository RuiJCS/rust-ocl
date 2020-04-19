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
        CLK_ADDRESS_MIRRORED_REPEAT |
        CLK_FILTER_NEAREST;

__kernel void blue(read_only image2d_t image, write_only image2d_t result) {
        int2 coord = (int2) (get_global_id(0),get_global_id(1));

        float4 pixel = read_imagef(image,sampler_const,coord);

        pixel += (float4) (0.0, 0.0, 0.2, 0.0);

        write_imagef(result,coord,pixel);
}

__kernel void convolute(read_only image2d_t src, write_only image2d_t result, __constant float4 * flt) {
        int filter_index = 0;
        float4 sum = (float4) 0.0;
        int2 coord = (int2) (get_global_id(0),get_global_id(1));


        for (int i = -KERNEL_SIZE_HALF; i <= KERNEL_SIZE_HALF; i++) {
                for(int j = -KERNEL_SIZE_HALF; j <= KERNEL_SIZE_HALF; j++, filter_index++) {
                        int2 local_coord = (int2)(coord.x + i,coord.y + j);
                        float4 pixel = read_imagef(src,sampler_const,local_coord);
                        sum += pixel * flt[filter_index];
                } 
        }

        write_imagef(result,coord,sum);
}


__kernel void convolute_mem(__read_only image2d_t src, __write_only image2d_t result, __constant float4 * filter) {
        int2 global_coord = (int2) (get_global_id(0),get_global_id(1));
        int2 local_coord = (int2) (get_local_id(0),get_local_id(1));
        int2 local_size = (int2) (get_local_size(0),get_local_size(1));
        int2 local_array = (int2) (local_coord.x + KERNEL_SIZE_HALF, local_coord.y + KERNEL_SIZE_HALF);
        
        // Need to think of the size of the array
        __local float4 pixels[BLOCK_SIZE][BLOCK_SIZE];

        pixels[local_array.y][local_array.x] = read_imagef(src,sampler_const,global_coord);

         // Places to load the image to local memory
        if(local_coord.x < KERNEL_SIZE_HALF) {
                // Left side of the kernel
                int2 local_mem = (int2) (local_array.x - KERNEL_SIZE_HALF, local_array.y);
                int2 global_mem = (int2) (global_coord.x - KERNEL_SIZE_HALF, global_coord.y);

                pixels[local_mem.y][local_mem.x] = read_imagef(src,sampler_const,global_mem);

                if(local_coord.y < KERNEL_SIZE_HALF) {
                        // Top left corner of the kernel
                        int2 local_mem = (int2) (local_array.x - KERNEL_SIZE_HALF, local_array.y - KERNEL_SIZE_HALF);
                        int2 global_mem = (int2) (global_coord.x - KERNEL_SIZE_HALF, global_coord.y - KERNEL_SIZE_HALF);

                        pixels[local_mem.y][local_mem.x] = read_imagef(src,sampler_const,global_mem);
                }

        }
        if(local_coord.x >= local_size.x - KERNEL_SIZE_HALF) {
                // Right side of the kernel
                int2 local_mem = (int2) (local_array.x + KERNEL_SIZE_HALF, local_array.y);
                int2 global_mem = (int2) (global_coord.x + KERNEL_SIZE_HALF, global_coord.y);

                pixels[local_mem.y][local_mem.x] = read_imagef(src,sampler_const,global_mem);
                if(local_coord.y >= local_size.y - KERNEL_SIZE_HALF) {
                        // Bottom right corner of the kernel
                        int2 local_mem = (int2) (local_array.x + KERNEL_SIZE_HALF, local_array.y + KERNEL_SIZE_HALF);
                        int2 global_mem = (int2) (global_coord.x + KERNEL_SIZE_HALF, global_coord.y + KERNEL_SIZE_HALF);

                        pixels[local_mem.y][local_mem.x] = read_imagef(src,sampler_const,global_mem);
                }
        }

        if(local_coord.y < KERNEL_SIZE_HALF) {
                // Top of the kernel
                int2 local_mem = (int2) (local_array.x, local_array.y - KERNEL_SIZE_HALF);
                int2 global_mem = (int2) (global_coord.x, global_coord.y - KERNEL_SIZE_HALF);

                pixels[local_mem.y][local_mem.x] = read_imagef(src,sampler_const,global_mem);

                if(local_coord.x >= local_size.x - KERNEL_SIZE_HALF) {
                        // Top right corner of the kernel
                        int2 local_mem = (int2) (local_array.x + KERNEL_SIZE_HALF, local_array.y - KERNEL_SIZE_HALF);
                        int2 global_mem = (int2) (global_coord.x + KERNEL_SIZE_HALF, global_coord.y - KERNEL_SIZE_HALF);

                        pixels[local_mem.y][local_mem.x] = read_imagef(src,sampler_const,global_mem);
                }
        }
        if(local_coord.y >= local_size.y - KERNEL_SIZE_HALF) {
                // Bottom of the kernel
                int2 local_mem = (int2) (local_array.x, local_array.y + KERNEL_SIZE_HALF);
                int2 global_mem = (int2) (global_coord.x, global_coord.y + KERNEL_SIZE_HALF);

                pixels[local_mem.y][local_mem.x] = read_imagef(src,sampler_const,global_mem);
                if(local_coord.x < KERNEL_SIZE_HALF) {
                        // Bottom left corner of the kernel
                        int2 local_mem = (int2) (local_array.x - KERNEL_SIZE_HALF, local_array.y + KERNEL_SIZE_HALF);
                        int2 global_mem = (int2) (global_coord.x - KERNEL_SIZE_HALF, global_coord.y + KERNEL_SIZE_HALF);

                        pixels[local_mem.y][local_mem.x] = read_imagef(src,sampler_const,global_mem);
                }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
        
        // Calculate Convolution
        int filter_index = 0;
        float4 sum = (float4) 0.0;
        for (int i = -KERNEL_SIZE_HALF; i <= KERNEL_SIZE_HALF; i++) {
                for(int j = -KERNEL_SIZE_HALF; j <= KERNEL_SIZE_HALF; j++, filter_index++) {
                        int2 local_mem = (int2)(local_array.x + j,local_array.y + i);
                        sum += pixels[local_mem.y][local_mem.x] * filter[filter_index];
                } 
        }

        // sum = pixels[local_array.y] [local_array.x];

        write_imagef(result,global_coord,sum);

}