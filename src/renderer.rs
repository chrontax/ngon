use std::{ffi::CStr, mem::size_of};

use crate::{
    buffer::Buffer,
    context::Context,
    polygon::{indices, vertices},
    utils::{
        choose_swap_extent, choose_swap_present_mode, choose_swap_surface_format,
        QueueFamilyIndices, SwapchainSupportDetails,
    },
};

use anyhow::Result;
use ash::{
    prelude::VkResult,
    vk::{self, PFN_vkBindImageMemory},
};
use inline_spirv::include_spirv as i_spirv;
use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::{ElementState, KeyEvent, WindowEvent},
    event_loop::ActiveEventLoop,
    keyboard::{Key, NamedKey},
    window::{Window, WindowId},
};

macro_rules! include_spirv {
    ($path:expr, $stage:ident) => {
        i_spirv!($path, $stage, glsl, entry = "main")
    };
}

const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;
const FRAGMENT_SHADER: &[u32] = include_spirv!("src/shader.frag", frag);
const VERTEX_SHADER: &[u32] = include_spirv!("src/shader.vert", vert);
const MAX_FRAMES_IN_FLIGHT: usize = 2;

#[derive(Default)]
pub struct Renderer {
    window: Option<Window>,
    ctx: Option<Context>,

    surface: vk::SurfaceKHR,

    queue_indices: QueueFamilyIndices,
    graphics_queue: vk::Queue,
    present_queue: vk::Queue,

    swapchain: vk::SwapchainKHR,
    swapchain_images: Vec<vk::Image>,
    swapchain_image_format: vk::Format,
    swapchain_extent: vk::Extent2D,
    swapchain_image_views: Vec<vk::ImageView>,
    swapchain_framebuffers: Vec<vk::Framebuffer>,

    render_pass: vk::RenderPass,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,

    pub vertex_count: usize,
    vertex_buffer: Buffer,
    index_count: u32,
    index_buffer: Buffer,

    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,

    flight_fences: [vk::Fence; MAX_FRAMES_IN_FLIGHT],
    image_available_semaphores: [vk::Semaphore; MAX_FRAMES_IN_FLIGHT],
    render_finished_semaphores: [vk::Semaphore; MAX_FRAMES_IN_FLIGHT],

    current_frame: usize,
    resized: bool,
}

impl Renderer {
    fn init(&mut self) -> Result<()> {
        self.create_queues();
        self.create_swapchain()?;
        self.create_pipeline()?;
        self.create_framebuffers()?;
        self.create_cmd()?;
        self.create_vertex_buffer()?;
        self.create_index_buffer()?;
        self.create_sync()?;

        Ok(())
    }

    fn create_queues(&mut self) {
        unsafe {
            let ctx = self.ctx.as_ref().unwrap();
            self.queue_indices = QueueFamilyIndices::find(
                &ctx.instance,
                &ctx.surface_instance,
                self.surface,
                ctx.physical_device,
            );
            self.graphics_queue = ctx.device.get_device_queue(self.queue_indices.graphics, 0);
            self.present_queue = ctx.device.get_device_queue(self.queue_indices.present, 0);
        }
    }

    fn create_swapchain(&mut self) -> VkResult<()> {
        let ctx = self.ctx.as_ref().unwrap();
        let details = SwapchainSupportDetails::query(
            ctx.physical_device,
            &ctx.surface_instance,
            self.surface,
        )?;

        let surface_format = choose_swap_surface_format(&details.formats);
        let present_mode = choose_swap_present_mode(&details.present_modes);
        self.swapchain_extent = choose_swap_extent(
            &details.capabilities,
            self.window.as_ref().unwrap().inner_size(),
        );

        let mut image_count = details.capabilities.min_image_count + 1;

        if details.capabilities.max_image_count > 0
            && image_count > details.capabilities.max_image_count
        {
            image_count = details.capabilities.max_image_count
        }

        let mut swapchain_ci = vk::SwapchainCreateInfoKHR::default()
            .surface(self.surface)
            .min_image_count(image_count)
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_extent(self.swapchain_extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .pre_transform(details.capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true);

        let indices = [self.queue_indices.graphics, self.queue_indices.present];
        if self.queue_indices.graphics != self.queue_indices.present {
            swapchain_ci = swapchain_ci
                .image_sharing_mode(vk::SharingMode::CONCURRENT)
                .queue_family_indices(&indices);
        }

        self.swapchain = unsafe { ctx.swapchain_device.create_swapchain(&swapchain_ci, None)? };
        self.swapchain_images =
            unsafe { ctx.swapchain_device.get_swapchain_images(self.swapchain)? };
        self.swapchain_image_format = surface_format.format;

        self.swapchain_image_views = self
            .swapchain_images
            .iter()
            .map(|&img| unsafe {
                ctx.device.create_image_view(
                    &vk::ImageViewCreateInfo::default()
                        .image(img)
                        .format(self.swapchain_image_format)
                        .components(vk::ComponentMapping {
                            r: vk::ComponentSwizzle::IDENTITY,
                            g: vk::ComponentSwizzle::IDENTITY,
                            b: vk::ComponentSwizzle::IDENTITY,
                            a: vk::ComponentSwizzle::IDENTITY,
                        })
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: 0,
                            layer_count: 1,
                        })
                        .view_type(vk::ImageViewType::TYPE_2D),
                    None,
                )
            })
            .collect::<VkResult<_>>()?;

        Ok(())
    }

    fn create_pipeline(&mut self) -> VkResult<()> {
        let ctx = self.ctx.as_ref().unwrap();

        self.pipeline_layout = unsafe {
            ctx.device
                .create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default(), None)?
        };

        let color_attachment = [vk::AttachmentDescription::default()
            .format(self.swapchain_image_format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)];

        let subpass = [vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&[vk::AttachmentReference {
                attachment: 0,
                layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            }])];

        let dependency = [vk::SubpassDependency::default()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags::NONE)
            .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)];

        self.render_pass = unsafe {
            ctx.device.create_render_pass(
                &vk::RenderPassCreateInfo::default()
                    .attachments(&color_attachment)
                    .subpasses(&subpass)
                    .dependencies(&dependency),
                None,
            )?
        };

        let vertex_shader_module = ctx.create_shader_module(VERTEX_SHADER)?;
        let fragment_shader_module = ctx.create_shader_module(FRAGMENT_SHADER)?;

        let stages = unsafe {
            [
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::VERTEX)
                    .module(vertex_shader_module)
                    .name(CStr::from_bytes_with_nul_unchecked(b"main\0")),
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::FRAGMENT)
                    .module(fragment_shader_module)
                    .name(CStr::from_bytes_with_nul_unchecked(b"main\0")),
            ]
        };

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state_ci =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

        let attributes =
            [vk::VertexInputAttributeDescription::default().format(vk::Format::R32G32_SFLOAT)];
        let bindings = [vk::VertexInputBindingDescription::default()
            .stride(size_of::<f32>() as u32 * 2)
            .input_rate(vk::VertexInputRate::VERTEX)];
        let vertex_input_state_ci = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_attribute_descriptions(&attributes)
            .vertex_binding_descriptions(&bindings);
        let input_assembly_state_ci = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        let viewport_state_ci = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);
        let rasterization_state_ci = vk::PipelineRasterizationStateCreateInfo::default()
            .depth_clamp_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.);
        let multisample_state_ci = vk::PipelineMultisampleStateCreateInfo::default()
            .sample_shading_enable(false)
            .rasterization_samples(vk::SampleCountFlags::TYPE_1)
            .min_sample_shading(1.);
        let color_blend_state_ci = vk::PipelineColorBlendStateCreateInfo::default().attachments(&[
            vk::PipelineColorBlendAttachmentState {
                color_write_mask: vk::ColorComponentFlags::RGBA,
                blend_enable: vk::TRUE,
                src_color_blend_factor: vk::BlendFactor::SRC_ALPHA,
                dst_color_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
                color_blend_op: vk::BlendOp::ADD,
                src_alpha_blend_factor: vk::BlendFactor::ONE,
                dst_alpha_blend_factor: vk::BlendFactor::ZERO,
                alpha_blend_op: vk::BlendOp::ADD,
            },
        ]);

        self.pipeline = match unsafe {
            ctx.device.create_graphics_pipelines(
                vk::PipelineCache::null(),
                &[vk::GraphicsPipelineCreateInfo::default()
                    .stages(&stages)
                    .vertex_input_state(&vertex_input_state_ci)
                    .input_assembly_state(&input_assembly_state_ci)
                    .viewport_state(&viewport_state_ci)
                    .rasterization_state(&rasterization_state_ci)
                    .multisample_state(&multisample_state_ci)
                    .color_blend_state(&color_blend_state_ci)
                    .dynamic_state(&dynamic_state_ci)
                    .layout(self.pipeline_layout)
                    .render_pass(self.render_pass)
                    .subpass(0)],
                None,
            )
        } {
            Ok(p) => p,
            Err((p, r)) => {
                r.result()?;
                p
            }
        }[0];

        unsafe {
            ctx.device.destroy_shader_module(vertex_shader_module, None);
            ctx.device
                .destroy_shader_module(fragment_shader_module, None);
        }

        Ok(())
    }

    fn create_framebuffers(&mut self) -> VkResult<()> {
        let ctx = self.ctx.as_ref().unwrap();

        self.swapchain_framebuffers = self
            .swapchain_image_views
            .iter()
            .map(|&view| unsafe {
                ctx.device.create_framebuffer(
                    &vk::FramebufferCreateInfo::default()
                        .render_pass(self.render_pass)
                        .attachment_count(1)
                        .attachments(&[view])
                        .width(self.swapchain_extent.width)
                        .height(self.swapchain_extent.height)
                        .layers(1),
                    None,
                )
            })
            .collect::<VkResult<_>>()?;

        Ok(())
    }

    fn create_cmd(&mut self) -> VkResult<()> {
        let ctx = self.ctx.as_ref().unwrap();

        self.command_pool = unsafe {
            ctx.device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                    .queue_family_index(self.queue_indices.graphics),
                None,
            )?
        };

        self.command_buffers = unsafe {
            ctx.device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(self.command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(MAX_FRAMES_IN_FLIGHT as u32),
            )?
        };

        Ok(())
    }

    fn create_vertex_buffer(&mut self) -> Result<()> {
        let ctx = self.ctx.as_ref().unwrap();

        let buffer_size = (size_of::<f32>() * self.vertex_count * 2) as _;

        let staging_buffer = Buffer::new(
            ctx,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            buffer_size,
        )?;

        self.vertex_buffer = Buffer::new(
            ctx,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            buffer_size,
        )?;

        staging_buffer.copy_from(
            vertices(self.vertex_count).as_ptr() as _,
            buffer_size as _,
            ctx,
        )?;

        staging_buffer.copy_to(
            &self.vertex_buffer,
            ctx,
            self.command_pool,
            self.graphics_queue,
        )?;
        staging_buffer.free(ctx);

        Ok(())
    }

    fn create_index_buffer(&mut self) -> Result<()> {
        let ctx = self.ctx.as_ref().unwrap();

        let indices = indices(self.vertex_count);
        let buffer_size = (size_of::<u16>() * indices.len()) as _;

        self.index_count = indices.len() as _;

        let staging_buffer = Buffer::new(
            ctx,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            buffer_size,
        )?;

        self.index_buffer = Buffer::new(
            ctx,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            buffer_size,
        )?;

        staging_buffer.copy_from(indices.as_ptr() as _, buffer_size as _, ctx)?;

        staging_buffer.copy_to(
            &self.index_buffer,
            ctx,
            self.command_pool,
            self.graphics_queue,
        )?;
        staging_buffer.free(ctx);

        Ok(())
    }

    fn create_sync(&mut self) -> VkResult<()> {
        let ctx = self.ctx.as_ref().unwrap();

        unsafe {
            for i in 0..MAX_FRAMES_IN_FLIGHT {
                self.image_available_semaphores[i] =
                    ctx.device.create_semaphore(&Default::default(), None)?;
                self.render_finished_semaphores[i] =
                    ctx.device.create_semaphore(&Default::default(), None)?;
                self.flight_fences[i] = ctx.device.create_fence(
                    &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                    None,
                )?;
            }
        }

        Ok(())
    }

    fn cleanup_swapchain(&self) {
        let ctx = self.ctx.as_ref().unwrap();

        unsafe {
            ctx.swapchain_device.destroy_swapchain(self.swapchain, None);

            for &framebuffer in &self.swapchain_framebuffers {
                ctx.device.destroy_framebuffer(framebuffer, None);
            }
            for &view in &self.swapchain_image_views {
                ctx.device.destroy_image_view(view, None);
            }
        }
    }

    fn recreate_swapchain(&mut self) -> VkResult<()> {
        unsafe {
            self.ctx.as_ref().unwrap().device.device_wait_idle()?;
        }

        self.cleanup_swapchain();

        self.create_swapchain()?;
        self.create_framebuffers()
    }

    fn draw(&mut self) -> VkResult<()> {
        let command_buffer = self.command_buffers[self.current_frame];
        let flight_fence = self.flight_fences[self.current_frame];
        let image_available_semaphore = self.image_available_semaphores[self.current_frame];
        let render_finished_semaphore = self.render_finished_semaphores[self.current_frame];
        let ctx = self.ctx.as_ref().unwrap();

        unsafe {
            ctx.device
                .wait_for_fences(&[flight_fence], true, u64::MAX)?;

            let image_index = if self.resized {
                self.resized = false;
                return self.recreate_swapchain();
            } else {
                match ctx.swapchain_device.acquire_next_image(
                    self.swapchain,
                    u64::MAX,
                    image_available_semaphore,
                    vk::Fence::null(),
                ) {
                    Ok((i, _)) => i,
                    Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => return self.recreate_swapchain(),
                    other => other?.0,
                }
            };

            ctx.device.reset_fences(&[flight_fence]).unwrap();

            ctx.device
                .reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())?;

            ctx.device
                .begin_command_buffer(command_buffer, &Default::default())?;

            ctx.device.cmd_begin_render_pass(
                command_buffer,
                &vk::RenderPassBeginInfo::default()
                    .render_pass(self.render_pass)
                    .framebuffer(self.swapchain_framebuffers[image_index as usize])
                    .render_area(vk::Rect2D::default().extent(self.swapchain_extent))
                    .clear_values(&[vk::ClearValue {
                        color: vk::ClearColorValue {
                            float32: [0., 0., 0., 1.],
                        },
                    }]),
                vk::SubpassContents::INLINE,
            );

            ctx.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline,
            );

            ctx.device.cmd_set_viewport(
                command_buffer,
                0,
                &[vk::Viewport {
                    x: 0.,
                    y: 0.,
                    width: self.swapchain_extent.width as f32,
                    height: self.swapchain_extent.height as f32,
                    min_depth: 0.,
                    max_depth: 1.,
                }],
            );

            ctx.device.cmd_set_scissor(
                command_buffer,
                0,
                &[vk::Rect2D::default().extent(self.swapchain_extent)],
            );

            ctx.device.cmd_bind_vertex_buffers(
                command_buffer,
                0,
                &[self.vertex_buffer.buffer],
                &[0],
            );
            ctx.device.cmd_bind_index_buffer(
                command_buffer,
                self.index_buffer.buffer,
                0,
                vk::IndexType::UINT16,
            );

            ctx.device
                .cmd_draw_indexed(command_buffer, self.index_count, 1, 0, 0, 0);
            ctx.device.cmd_end_render_pass(command_buffer);

            ctx.device.end_command_buffer(command_buffer).unwrap();

            let signal_semaphores = [render_finished_semaphore];

            ctx.device.queue_submit(
                self.graphics_queue,
                &[vk::SubmitInfo::default()
                    .wait_dst_stage_mask(&[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT])
                    .wait_semaphores(&[image_available_semaphore])
                    .command_buffers(&[command_buffer])
                    .signal_semaphores(&signal_semaphores)],
                flight_fence,
            )?;

            ctx.swapchain_device.queue_present(
                self.present_queue,
                &vk::PresentInfoKHR::default()
                    .wait_semaphores(&signal_semaphores)
                    .swapchains(&[self.swapchain])
                    .image_indices(&[image_index]),
            )?;
        }

        self.current_frame = (self.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;

        Ok(())
    }
}

impl ApplicationHandler for Renderer {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let mut window = None;
        while let None | Some(Err(_)) = window {
            window = Some(
                event_loop.create_window(
                    Window::default_attributes()
                        .with_inner_size(LogicalSize::new(WIDTH as f64, HEIGHT as _)),
                ),
            );
        }

        self.window = Some(window.unwrap().unwrap());
        self.ctx = Some(Context::new(self.window.as_ref().unwrap(), &mut self.surface).unwrap());

        self.init().unwrap();
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested
            | WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: ElementState::Pressed,
                        logical_key: Key::Named(NamedKey::Escape),
                        ..
                    },
                ..
            } => event_loop.exit(),
            WindowEvent::Resized(_) => self.resized = true,
            WindowEvent::RedrawRequested => {
                self.draw().unwrap();
                self.window.as_ref().unwrap().request_redraw();
            }
            _ => (),
        }
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        let ctx = self.ctx.as_ref().unwrap();

        unsafe {
            ctx.device.device_wait_idle().unwrap();

            self.vertex_buffer.free(ctx);
            self.index_buffer.free(ctx);

            for &semaphore in self
                .image_available_semaphores
                .iter()
                .chain(self.render_finished_semaphores.iter())
            {
                ctx.device.destroy_semaphore(semaphore, None);
            }
            for &fence in &self.flight_fences {
                ctx.device.destroy_fence(fence, None);
            }

            self.cleanup_swapchain();

            ctx.device.destroy_command_pool(self.command_pool, None);
            ctx.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            ctx.device.destroy_pipeline(self.pipeline, None);
            ctx.device.destroy_render_pass(self.render_pass, None);

            ctx.surface_instance.destroy_surface(self.surface, None);
        }
    }
}
