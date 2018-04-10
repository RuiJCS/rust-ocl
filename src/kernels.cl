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


__kernel void convolute_mem(read_only image2d_t src, write_only image2d_t result, __global float4 * flt) {
        int fIndex = 0;
        float4 sum = (float4) 0.0;
        int2 coord = (int2) (get_global_id(0),get_global_id(1));
        int2 local_coord = (int2) (get_local_id(0),get_local_id(1));
        int local_array = local_coord.x * ROW_SIZE + local_coord.y;
        

        __local float4 * pixels;

        pixels = read_imagef(src,sampler_const,local_coord);

        barrier(CLK_LOCAL_MEM_FENCE);

        // for (int i = 0; i <= fIndex; i++) {
        //         sum += pixels[i] * flt[i];
        // }
        // sum = (float4) (0.0,0.0,1.0,0.0);

        write_imagef(result,coord,sum);

}