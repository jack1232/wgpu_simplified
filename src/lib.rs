use winit::window::Window;
//use std::path::PathBuf;
use std::f32::consts::PI;
use cgmath::*;

#[rustfmt::skip]
#[allow(unused)]

// region: bind groups

pub fn create_bind_group_layout(
    device: &wgpu::Device, 
    shader_stages: Vec<wgpu::ShaderStages>
) -> wgpu::BindGroupLayout {
    let mut entries = vec![];
    
    for i in 0..shader_stages.len() {
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: i as u32,
            visibility: shader_stages[i],
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });
    }
    
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor{
        entries: &entries,
        label: Some("Uniform Bind Group Layout"),
    })
}

pub fn create_bind_group(
    device: &wgpu::Device,
    shader_stages: Vec<wgpu::ShaderStages>,
    resources: &[wgpu::BindingResource<'_>]
) -> ( wgpu::BindGroupLayout, wgpu::BindGroup) {
    let entries: Vec<_> = resources.iter().enumerate().map(|(i, resource)| {
        wgpu::BindGroupEntry {
            binding: i as u32,
            resource: resource.clone(),
        }
    }).collect();

    let layout = create_bind_group_layout(device, shader_stages);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &layout,
        entries: &entries,
        label: Some("Uniform Bind Group"),
    });

    (layout, bind_group)
}

// endregion: bind groups


// region: tranformation
pub const OPENGL_TO_WGPU_MATRIX: Matrix4<f32> = Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.0,
    0.0, 0.0, 0.5, 1.0,
);

pub fn create_model_mat(translation:[f32; 3], rotation:[f32; 3], scaling:[f32; 3]) -> Matrix4<f32> {
    // create transformation matrices
    let trans_mat = Matrix4::from_translation(Vector3::new(translation[0], translation[1], translation[2]));
    let rotate_mat_x = Matrix4::from_angle_x(Rad(rotation[0]));
    let rotate_mat_y = Matrix4::from_angle_y(Rad(rotation[1]));
    let rotate_mat_z = Matrix4::from_angle_z(Rad(rotation[2]));
    let scale_mat = Matrix4::from_nonuniform_scale(scaling[0], scaling[1], scaling[2]);

    // combine all transformation matrices together to form a final transform matrix: model matrix
    let model_mat = trans_mat * rotate_mat_z * rotate_mat_y * rotate_mat_x * scale_mat;

    // return final model matrix
    model_mat
}

pub fn create_view_mat(camera_position: Point3<f32>, look_direction: Point3<f32>, up_direction: Vector3<f32>) -> Matrix4<f32> {
    Matrix4::look_at_rh(camera_position, look_direction, up_direction)
}

pub fn create_projection_mat(aspect:f32, is_perspective:bool) -> Matrix4<f32> {
    let project_mat:Matrix4<f32>;
    if is_perspective {
        project_mat = OPENGL_TO_WGPU_MATRIX * perspective(Rad(2.0*PI/5.0), aspect, 0.1, 1000.0);
    } else {
        project_mat = OPENGL_TO_WGPU_MATRIX * ortho(-4.0, 4.0, -3.0, 3.0, -1.0, 6.0);
    }
    project_mat
}

pub fn create_perspective_mat(fovy:Rad<f32>, aspect:f32, near: f32, far:f32) -> Matrix4<f32> {
    OPENGL_TO_WGPU_MATRIX * perspective(fovy, aspect, near, far)
}

pub fn create_ortho_mat(left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32) -> Matrix4<f32> {
    OPENGL_TO_WGPU_MATRIX * ortho(left, right, bottom, top, near, far)    
}

pub fn create_vp_ortho_mat(left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32, camera_position: Point3<f32>, 
    look_direction: Point3<f32>, up_direction: Vector3<f32>) -> (Matrix4<f32>, Matrix4<f32>, Matrix4<f32>) {
    
    // construct view matrix
    let view_mat = Matrix4::look_at_rh(camera_position, look_direction, up_direction);     

    // construct projection matrix
    let project_mat = OPENGL_TO_WGPU_MATRIX * ortho(left, right, bottom, top, near, far);    
    
    // contruct view-projection matrix
    let vp_mat = project_mat * view_mat;
   
    // return various matrices
    (view_mat, project_mat, vp_mat)
} 

pub fn create_vp_mat(camera_position: Point3<f32>, look_direction: Point3<f32>, up_direction: Vector3<f32>,
    aspect:f32) -> (Matrix4<f32>, Matrix4<f32>, Matrix4<f32>) {
    
    // construct view matrix
    let view_mat = Matrix4::look_at_rh(camera_position, look_direction, up_direction);     

    // construct projection matrix
    let project_mat = OPENGL_TO_WGPU_MATRIX * perspective(Rad(2.0*PI/5.0), aspect, 0.1, 1000.0);
   
    // contruct view-projection matrix
    let vp_mat = project_mat * view_mat;
   
    // return various matrices
    (view_mat, project_mat, vp_mat)
} 

pub fn create_projection_ortho(left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32) -> Matrix4<f32> {
    OPENGL_TO_WGPU_MATRIX * ortho(left, right, bottom, top, near, far)    
}

// endregion: tranformation


// region: views and attachments
pub fn create_color_attachment<'a>(texture_view: &'a wgpu::TextureView) -> wgpu::RenderPassColorAttachment<'a> {
    wgpu::RenderPassColorAttachment {
        view: texture_view,
        resolve_target: None,
        ops: wgpu::Operations {
            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
            store: true,
        },
    }
}

pub fn create_msaa_texture_view(init: &IWgpuInit) -> wgpu::TextureView{
    let msaa_texture = init.device.create_texture(&wgpu::TextureDescriptor {
        size: wgpu::Extent3d {
            width: init.config.width,
            height: init.config.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: init.sample_count,
        dimension: wgpu::TextureDimension::D2,
        format: init.config.format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        label: None,
        view_formats: &[],
    });
    
    msaa_texture.create_view(&wgpu::TextureViewDescriptor::default())
}

pub fn create_msaa_color_attachment<'a>(texture_view: &'a wgpu::TextureView, msaa_view: &'a wgpu::TextureView) 
-> wgpu::RenderPassColorAttachment<'a> {
    wgpu::RenderPassColorAttachment {
        view: msaa_view,
        resolve_target: Some(texture_view),
        ops: wgpu::Operations {
            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
            store: true,
        },
    }
}

pub fn create_depth_view(init: &IWgpuInit) -> wgpu::TextureView {
    let depth_texture = init.device.create_texture(&wgpu::TextureDescriptor {
        size: wgpu::Extent3d {
            width: init.config.width,
            height: init.config.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: init.sample_count,
        dimension: wgpu::TextureDimension::D2,
        format:wgpu::TextureFormat::Depth24Plus,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        label: None,
        view_formats: &[],
    });
    
    depth_texture.create_view(&wgpu::TextureViewDescriptor::default())
}

pub fn create_depth_stencil_attachment<'a>(depth_view: &'a wgpu::TextureView) -> wgpu::RenderPassDepthStencilAttachment<'a> {
    wgpu::RenderPassDepthStencilAttachment {
        view: depth_view,
        depth_ops: Some(wgpu::Operations {
            load: wgpu::LoadOp::Clear(1.0),
            store: false,
        }),
        stencil_ops: None,
    }
}

// endregion: views and attachments


// region: pipelines
pub struct IRenderPipeline<'a> {
    pub shader: Option<&'a wgpu::ShaderModule>,
    pub vs_shader: Option<&'a wgpu::ShaderModule>,
    pub fs_shader: Option<&'a wgpu::ShaderModule>,
    pub vertex_buffer_layout: &'a [wgpu::VertexBufferLayout<'a>],
    pub pipeline_layout: Option<&'a wgpu::PipelineLayout>,
    pub topology: wgpu::PrimitiveTopology,
    pub strip_index_format: Option<wgpu::IndexFormat>,
    pub cull_mode: Option<wgpu::Face>,
    pub is_depth_stencil: bool,
    pub vs_entry: String,
    pub fs_entry: String,
}

impl Default for IRenderPipeline<'_> {
    fn default() -> Self {
        Self { 
            shader: None, 
            vs_shader: None,
            fs_shader: None,
            vertex_buffer_layout: &[],
            pipeline_layout: None,
            topology: wgpu::PrimitiveTopology::TriangleList, 
            strip_index_format: None, 
            cull_mode: None, 
            is_depth_stencil: true, 
            vs_entry: String::from("vs_main"), 
            fs_entry: String::from("fs_main"),            
        }
    }
}

impl IRenderPipeline<'_> {    
    pub fn new(&mut self, init: &IWgpuInit) -> wgpu::RenderPipeline {
        if self.shader.is_some() {
            self.vs_shader = self.shader;
            self.fs_shader = self.shader;
        }

        let mut depth_stencil:Option<wgpu::DepthStencilState> = None;
        if self.is_depth_stencil {
            depth_stencil = Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth24Plus,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            });
        } 

        init.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&self.pipeline_layout.unwrap()),
            vertex: wgpu::VertexState {
                module: &self.vs_shader.as_ref().unwrap(),
                entry_point: &self.vs_entry,
                buffers: &self.vertex_buffer_layout,
            },
            fragment: Some(wgpu::FragmentState {
                module: &self.fs_shader.as_ref().unwrap(),
                entry_point: &self.fs_entry,
                targets: &[Some(init.config.format.into())],
            }),
            primitive: wgpu::PrimitiveState {
                topology: self.topology,
                strip_index_format: self.strip_index_format,
                ..Default::default()
            },
            depth_stencil,
            multisample: wgpu::MultisampleState{
                count: init.sample_count,
                ..Default::default()
            },
            multiview: None,
        })
    }   
}

// endregion: pipelines


// region: wgpu initialization
pub struct IWgpuInit {
    pub instance: wgpu::Instance,
    pub surface: wgpu::Surface,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub config: wgpu::SurfaceConfiguration,
    pub size: winit::dpi::PhysicalSize<u32>,
    pub sample_count: u32,
}

impl IWgpuInit {
    pub async fn new(window: &Window, sample_count:u32, limits:Option<wgpu::Limits>) -> Self {
        let limits_device = limits.unwrap_or(wgpu::Limits::default());

        let size = window.inner_size();
        let instance = wgpu::Instance::default();
        /*let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::DX12,        
            dx12_shader_compiler: {
                wgpu::Dx12Compiler::Dxc { 
                    dxil_path: Some(PathBuf::from(r"assets/dxil.dll")), 
                    dxc_path: Some(PathBuf::from(r"assets/dxcompiler.dll")),
                }
            }
        });*/
    
        let surface = unsafe { instance.create_surface(&window) }.unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("Failed to find an appropriate adapter");
              
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    //features: wgpu::Features::empty(),
                    features:wgpu::Features::default() | wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES,
                    //limits: wgpu::Limits::default(),
                    limits: limits_device
                },
                None,
            )
            .await
            .expect("Failed to create device");
        
        let surface_caps = surface.get_capabilities(&adapter);
        let format = surface_caps.formats[0];
            
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode:surface_caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&device, &config);

        Self {
            instance,
            surface,
            adapter,
            device,
            queue,
            config,
            size,
            sample_count,
        }    
    }
}

// endregion: wgpu initialization