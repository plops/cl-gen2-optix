(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2")
  (ql:quickload "cl-ppcre"))

(in-package :cl-cpp-generator2)

(setf *features* (union *features* '(:nolog
				     :debug-thread-activity
				     :serial-debug
				     :queue-debug
				     :lock-debug
				     
				     )))
(setf *features* (set-difference *features*
				 '(:nolog
				   :debug-thread-activity
				   :serial-debug
				   :queue-debug
				   
				   )))


(progn
  ;; make sure to run this code twice during the first time, so that
  ;; the functions are defined

  (defparameter *source-dir* #P"../cl-gen2-optix/source/")

  (progn
    ;; collect code that will be emitted in utils.h
    (defparameter *utils-code* nil)
    (defun emit-utils (&key code)
      (push code *utils-code*)
      " "))
  (progn
    (defparameter *module-global-parameters* nil)
    (defparameter *module* nil)

    (defun set-members (params)
        "setf on multiple member variables of an instance"
        (destructuring-bind (instance &rest args) params
          `(setf ,@(loop for i from 0 below (length args) by 2 appending
                        (let ((keyword (elt args i))
                              (value (elt args (+ i 1))))
                          `((dot ,instance ,keyword) ,value))))))

    (defun set-members-clear (params)
        "setf on multiple member variables of an instance"
        (destructuring-bind (instance &rest args) params
          `(do0
	    (setf ,instance (curly))
	    (setf
	     ,@(loop for i from 0 below (length args) by 2 appending
		    (let ((keyword (elt args i))
			  (value (elt args (+ i 1))))
		      `((dot ,instance ,keyword) ,value)))))))

    (defun ox (code)
      `(progn
	 (let ((res ,code))
	  (declare (type OptixResult res))
	  (unless (== OPTIX_SUCCESS res)
	    ,(logprint (format nil (string "FAIL: optix ~a")
			       (cl-cpp-generator2::emit-c :code code))
		       `(res))
	    ))))
    (defun cu (code)
      `(progn
	 (let ((res ,code))
	   (unless (== cudaSuccess ;; CUDA_SUCCESS
		       res
		       )
	     (let ((err_ (cudaGetLastError))
		   (err_name (cudaGetErrorName err_))
		   (err_str (cudaGetErrorString err_)))
	      ,(logprint (format nil (string "FAIL: cuda ~a")
				 (cl-cpp-generator2::emit-c :code code))
			 `(res err_ err_name err_str)))
	    (throw ("std::runtime_error" (string ,(format nil (string "~a")
						   (cl-cpp-generator2::emit-c :code code)))))))))
    (defun logprint (msg &optional rest)
      `(do0
	" "
	#-nolog
	(do0
	 ;("std::setprecision" 3)
	 (<< "std::cout"
	     ;;"std::endl"
	     ("std::setw" 10)
	     (dot ("std::chrono::high_resolution_clock::now")
		  (time_since_epoch)
		  (count))
					;,(g `_start_time)
	     
	     (string " ")
	     ("std::this_thread::get_id")
	     (string " ")
	     __FILE__
	     (string ":")
	     __LINE__
	     (string " ")
	     __func__
	     (string " ")
	     (string ,msg)
	     (string " ")
	     ,@(loop for e in rest appending
		    `(("std::setw" 8)
					;("std::width" 8)
		      (string ,(format nil " ~a=" (emit-c :code e)))
		      ,e))
	     "std::endl"
	     "std::flush"))))
    (defun guard (code &key (debug t))
		  `(do0
		    #+lock-debug ,(if debug
		       (logprint (format nil "hold guard on ~a" (cl-cpp-generator2::emit-c :code code))
				 `())
		       "// no debug")
		    #+eou ,(if debug
		     `(if (dot ,code ("std::mutex::try_lock"))
			 (do0
			  (dot ,code (unlock)))
			 (do0
			  ,(logprint (format nil "have to wait on ~a" (cl-cpp-generator2::emit-c :code code))
				     `())))
		     "// no debug")
		    "// no debug"
		   ,(format nil
			    "std::lock_guard<std::mutex> guard(~a);"
			    (cl-cpp-generator2::emit-c :code code))))
    (defun lock (code &key (debug t))
      `(do0
	#+lock-debug ,(if debug
	     (logprint (format nil "hold lock on ~a" (cl-cpp-generator2::emit-c :code code))
		       `())
	     "// no debug")

	#+nil (if (dot ,code ("std::mutex::try_lock"))
	    (do0
	     (dot ,code (unlock)))
	    (do0
	     ,(logprint (format nil "have to wait on ~a" (cl-cpp-generator2::emit-c :code code))
			`())))
	
		    ,(format nil
			     "std::unique_lock<std::mutex> lk(~a);"
			     
				(cl-cpp-generator2::emit-c :code code))
		    ))

    
    (defun emit-globals (&key init)
      (let ((l `((_start_time ,(emit-c :code `(typeof (dot ("std::chrono::high_resolution_clock::now")
							   (time_since_epoch)
							   (count)))))
		 ,@(loop for e in *module-global-parameters* collect
			(destructuring-bind (&key name type default)
			    e
			  `(,name ,type))))))
	(if init
	    `(curly
	      ,@(remove-if
		 #'null
		 (loop for e in l collect
		      (destructuring-bind (name type &optional value) e
			(when value
			  `(= ,(format nil ".~a" (elt (cl-ppcre:split "\\[" (format nil "~a" name)) 0)) ,value))))))
	    `(do0
	      (include <chrono>)
	      (defstruct0 State
		  ,@(loop for e in l collect
 			 (destructuring-bind (name type &optional value) e
			   `(,name ,type))))))))
    (defun define-module (args)
      "each module will be written into a c file with module-name. the global-parameters the module will write to will be specified with their type in global-parameters. a file global.h will be written that contains the parameters that were defined in all modules. global parameters that are accessed read-only or have already been specified in another module need not occur in this list (but can). the prototypes of functions that are specified in a module are collected in functions.h. i think i can (ab)use gcc's warnings -Wmissing-declarations to generate this header. i split the code this way to reduce the amount of code that needs to be recompiled during iterative/interactive development. if the module-name contains vulkan, include vulkan headers. if it contains glfw, include glfw headers."
      (destructuring-bind (module-name global-parameters module-code) args
	(let ((header ()))
	  (push `(do0
		  " "
		  (include "utils.h")
		  " "
		  (include "globals.h")
		  " "
		  (include "proto2.h")
		  " ")
		header)
	  (unless (cl-ppcre:scan "main" (string-downcase (format nil "~a" module-name)))
	    (push `(do0 "extern State state;")
		  header))
	  (push `(:name ,module-name :code (do0 ,@(reverse header) ,module-code))
		*module*))
	(loop for par in global-parameters do
	     (destructuring-bind (parameter-name
				  &key (direction 'in)
				  (type 'int)
				  (default nil)) par
	       (push `(:name ,parameter-name :type ,type :default ,default)
		     *module-global-parameters*))))))
  (defun g (arg)
    `(dot state ,arg))
  
  (define-module
      `(main ((_filename :direction 'out :type "char const *")
	      (_pixels :type "std::vector<uint32_t>")
	      )
	     (do0
	      (include <iostream>
		       <chrono>
		       <cstdio>
		       <cassert>
					;<unordered_map>
		       <string>
		       <fstream>)

	      
	      (let ((state ,(emit-globals :init t)))
		(declare (type "State" state)))


	      (do0
	       (defun mainLoop ()
		 ,(logprint "mainLoop" `())

		 (let ((first_run true)
		       (model)
		       (camera (curly ("glm::vec3" -10s0 2s0 -12s0)
				      ("glm::vec3" 0s0 0s0 0s0)
				      ("glm::vec3" 0s0 1s0 0s0))))
		   (declare (type "static triangle_mesh_t" model)
			    (type "static camera_t" camera)
			    (type "static bool" first_run))

		   
		   (when first_run
		     (do0 (model.add_cube ("glm::vec3" 0s0 -1.5s0 0s0)
					("glm::vec3" 10s0 .1s0 10s0))
			(model.add_cube ("glm::vec3" 0s0 0s0 0s0)
					("glm::vec3" 2s0 2s0 2s0)))
		    (initOptix model)))
		 
		 (while (not (glfwWindowShouldClose ,(g `_window)))
		   (progn
		    (when (or first_run ,(g `_framebufferResized))
		      (let ((width 0)
			    (height 0))
			(declare (type int width height))
			
			(glfwGetWindowSize ,(g `_window)
					   &width
					   &height))
		      (do0 (resize width height)
			   (dot ,(g `_pixels)
				(resize (* width height))))
		      (setf ,(g `_framebufferResized) false
			    first_run false)))
		   (glfwPollEvents)
		   (drawFrame)
		   (do0
		    (render)
		    (download_pixels (dot ,(g `_pixels)
					  (data)))
		    (let ((fb_texture 0))
		      (declare (type "static GLuint" fb_texture))
		      (when (== 0 fb_texture)
			(glGenTextures 1 &fb_texture))
		      (glBindTexture GL_TEXTURE_2D fb_texture)
		      (glTexImage2D GL_TEXTURE_2D
				    0
				    GL_RGBA
				    ,(g `launch_params.fbSize_x)
				    ,(g `launch_params.fbSize_y)
				    0
				    GL_RGBA
				    GL_UNSIGNED_BYTE
				    (dot ,(g `_pixels)
					 (data)))
		      (glDisable GL_LIGHTING)
		      (glColor3f 1 1 1)
		      (glMatrixMode GL_MODELVIEW)
		      (glLoadIdentity)
		      (glEnable GL_TEXTURE_2D)
		      (glBindTexture GL_TEXTURE_2D fb_texture)
		      (glTexParameteri GL_TEXTURE_2D GL_TEXTURE_MIN_FILTER GL_LINEAR)
		      (glTexParameteri GL_TEXTURE_2D GL_TEXTURE_MAG_FILTER GL_LINEAR)
		      (glDisable GL_DEPTH_TEST)
		      (glViewport 0 0
				  ,(g `launch_params.fbSize_x)
				    ,(g `launch_params.fbSize_y)
				    )
		      (glMatrixMode GL_PROJECTION)
		      (glLoadIdentity)
		      (glOrtho 0s0
			       (static_cast<float> ,(g `launch_params.fbSize_x))
			       0s0
			       (static_cast<float> ,(g `launch_params.fbSize_y))
			       -1s0
			       1s0)
		      (do0
		       (glBegin GL_QUADS)
		       ,@(loop for (e f) in `((0 0)
					      (0 (static_cast<float> ,(g `launch_params.fbSize_y)))
					      ((static_cast<float> ,(g `launch_params.fbSize_x))
					       (static_cast<float> ,(g `launch_params.fbSize_y)))
					      ((static_cast<float> ,(g `launch_params.fbSize_x))
					       0))
			    appending
			      `((glTexCoord2f ,(if (eq e 0)
						   0s0
						   1s0)
					      ,(if (eq f 0)
						   0s0
						   1s0))
				(glVertex3f ,e ,f 0s0)))
		       (glEnd))))
		   (drawGui)
		   (glfwSwapBuffers ,(g `_window))
		   )
		 ,(logprint "exit mainLoop" `()))
	       (defun run ()
		 ,(logprint "start run" `())
		 
		 (initWindow)
		 (initGui)
		 
		 (initDraw)
		 
		 
		 (mainLoop)
		 ,(logprint "finish run" `())))
	      
	      (defun main ()
		(declare (values int))
		
		(setf ,(g `_start_time) (dot ("std::chrono::high_resolution_clock::now")
					     (time_since_epoch)
					     (count)))
		,(logprint "start main" `())
		(setf ,(g `_filename)
		      (string "bla.txt"))

		(do0
		 (run)
		 ,(logprint "start cleanups" `())
		 (cleanupOptix)
		 (cleanupDraw)
		 (cleanupGui)
		 (cleanupWindow)
		)
		,(logprint "end main" `())
		(return 0)))))

  
  
  
  (define-module
      `(glfw_window
	((_window :direction 'out :type GLFWwindow* )
	 (_framebufferResized :direction 'out :type bool)
	 )
	(do0
	 (defun keyCallback (window key scancode action mods)
	   (declare (type GLFWwindow* window)
		    (type int key scancode action mods))
	   (when (and (or (== key GLFW_KEY_ESCAPE)
			  (== key GLFW_KEY_Q))
		      (== action GLFW_PRESS))
	     (glfwSetWindowShouldClose window GLFW_TRUE)))
	 (defun errorCallback (err description)
	   (declare (type int err)
		    (type "const char*" description))
	   ,(logprint "error" `(err description)))
	 (defun framebufferResizeCallback (window width height)
	   (declare (values "static void")
		    ;; static because glfw doesnt know how to call a member function with a this pointer
		    (type GLFWwindow* window)
		    (type int width height))
	   ,(logprint "resize" `(width height))
	   #+nil (do0 (resize width height)
		(dot ,(g `_pixels)
		     (resize (* width height))))
	   (let ((app ("(State*)" (glfwGetWindowUserPointer window))))
	     (setf app->_framebufferResized true)))
	 (defun initWindow ()
	   (declare (values void))
	   (when (glfwInit)
	     (do0
	      (glfwSetErrorCallback errorCallback)
	      
	      (glfwWindowHint GLFW_CONTEXT_VERSION_MAJOR 2)
	      (glfwWindowHint GLFW_CONTEXT_VERSION_MINOR 0)
	      
	      (glfwWindowHint GLFW_RESIZABLE GLFW_TRUE)
	      (setf ,(g `_window) (glfwCreateWindow 930 930
						    (string "vis window")
						    NULL
						    NULL))
	      ,(logprint "initWindow" `(,(g `_window)
					 (glfwGetVersionString)))
	      ;; store this pointer to the instance for use in the callback
	      (glfwSetKeyCallback ,(g `_window) keyCallback)
	      (glfwSetWindowUserPointer ,(g `_window) (ref state))
	      (glfwSetFramebufferSizeCallback ,(g `_window)
					      framebufferResizeCallback)
	      (glfwMakeContextCurrent ,(g `_window))
	      (glfwSwapInterval 1)
	      )))
	 (defun cleanupWindow ()
	   (declare (values void))
	   (glfwDestroyWindow ,(g `_window))
	   (glfwTerminate)))))
  (define-module
      `(draw ((_fontTex :direction 'out :type GLuint))
	     (do0
	      (include <algorithm>)
	      (defun uploadTex (image w h)
		(declare (type "const void*" image)
			 (type int w h))
		(glGenTextures 1 (ref ,(g `_fontTex)))
		(glBindTexture GL_TEXTURE_2D ,(g `_fontTex))
		(glTexParameteri GL_TEXTURE_2D GL_TEXTURE_MIN_FILTER GL_LINEAR)
		(glTexParameteri GL_TEXTURE_2D GL_TEXTURE_MAG_FILTER GL_LINEAR)
		(glTexImage2D GL_TEXTURE_2D 0 GL_RGBA w h 0 GL_RGBA GL_UNSIGNED_BYTE image))
	      
	      (defun initDraw ()
					;(glEnable GL_TEXTURE_2D)
		#+nil (glEnable GL_DEPTH_TEST)
		#+nil (glHint GL_LINE_SMOOTH GL_NICEST)
		#+nil (do0 (glEnable GL_BLEND)
		     (glBlendFunc GL_SRC_ALPHA
				  GL_ONE_MINUS_SRC_ALPHA))
		(glClearColor 0 0 0 1))
	      (defun cleanupDraw ()
		(glDeleteTextures 1 (ref ,(g `_fontTex))))
	      (defun drawFrame ()
		(glClear (logior GL_COLOR_BUFFER_BIT
				 GL_DEPTH_BUFFER_BIT))))))
  (define-module
      `(gui ()
	    (do0
	     "// https://youtu.be/nVaQuNXueFw?t=317"
	     "// https://blog.conan.io/2019/06/26/An-introduction-to-the-Dear-ImGui-library.html"
	     (include "imgui/imgui.h"
		      "imgui/imgui_impl_glfw.h"
		      "imgui/imgui_impl_opengl2.h")
	     (include <algorithm>
		      <string>)
	     (defun initGui ()
	       ,(logprint "initGui" '())
	       (IMGUI_CHECKVERSION)
	       ("ImGui::CreateContext")
	       
	       (ImGui_ImplGlfw_InitForOpenGL ,(g `_window)
					     true)
	       (ImGui_ImplOpenGL2_Init)
	       ("ImGui::StyleColorsDark"))
	     (defun cleanupGui ()
	       (ImGui_ImplOpenGL2_Shutdown)
	       (ImGui_ImplGlfw_Shutdown)
	       ("ImGui::DestroyContext"))
	     
	     (defun drawGui ()
	       #+nil (<< "std::cout"
		   (string "g")
		   "std::flush")
	       
	       (ImGui_ImplOpenGL2_NewFrame)
	       (ImGui_ImplGlfw_NewFrame)
	       ("ImGui::NewFrame")
	       	       
	       (let ((b true))
		      ("ImGui::ShowDemoWindow" &b))
	       ("ImGui::Render")
	       (ImGui_ImplOpenGL2_RenderDrawData
		("ImGui::GetDrawData"))
	       ))))

  
  (define-module
      `(optix (
	       ;; createContext
	       (dev_id :type "int")
	       (stream :type CUstream)
	       (dev_prop :type cudaDeviceProp)
	       (cuctx :type CUcontext)
	       (oxctx :type OptixDeviceContext)
	       ;; createModule
	       (module_compile_options :type OptixModuleCompileOptions)
	       (pipeline_compile_options :type OptixPipelineCompileOptions)
	       (pipeline_link_options :type OptixPipelineLinkOptions)

	       ;;
	       (pipeline :type OptixPipeline)
	       (module :type OptixModule)
	       ;;

	       ,@(loop for e in `(raygen miss hitgroup)
		    appending
		      `((,(format nil "~a_programs" e) :type "std::vector<OptixProgramGroup>")
			(,(format nil "~a_records_buffer" e) :type CUDABuffer))
		      )
	       
	       ;(hitgroup_records_buffer :type CUDABuffer)
	       (shader_bindings_table :type OptixShaderBindingTable)
	       ;;
	       (launch_params :type LaunchParams)
	       (launch_params_buffer :type CUDABuffer)
	       ;;
	       (color_buffer :type CUDABuffer)
	       ;;
	       (last_set_camera :type camera_t)
	       ;;
	       (vertex_buffer :type CUDABuffer)
	       (index_buffer :type CUDABuffer)
	       ;;
	       (accel_buffer :type CUDABuffer)
	       
	       )
	      (do0
	       "// derived from Ingo Wald's optix7course example03_inGLFWindow SampleRenderer.cpp"
	       (include <cuda_runtime.h>
			<optix.h>
			<optix_stubs.h>
			<optix_function_table_definition.h>)
	       " "
	       (include <glm/geometric.hpp> )
					
	       "extern \"C\" const  char ptx_code[];"

	     (defun createContext ()
	       (declare (type "static void"))
	       (let ((count 0))
		 (declare (type int count))
		 ,(cu `(cudaGetDeviceCount &count))
		 ,(logprint "get device count" `(count)))
	       (setf ,(g `dev_id) 0)
	       ,(cu `(cudaSetDevice ,(g `dev_id)))
	       ,(cu `(cudaStreamCreate (ref ,(g `stream))))
	       (cudaGetDeviceProperties (ref ,(g `dev_prop))
					,(g `dev_id))
	       ,(logprint "running on device:"
			  `(,(g `dev_prop.name)))
	       ,(cu `(cuCtxGetCurrent (ref ,(g `cuctx))))
	       ,(ox
		 `(optixDeviceContextCreate
		   ,(g `cuctx) 0 (ref ,(g `oxctx))))
	       (let ((log_cb (lambda (level
			    tag
			    msg
			    data)
		     (declare
		      (type "unsigned int" level)
		      (type "const char*"
			    tag
			    msg)
		      (type "void*"
			    data))
		     ,(logprint "context_log"
				`(level
				  tag
				  msg)))))
		,(ox
		  `(optixDeviceContextSetLogCallback
		    ,(g `oxctx)
		   log_cb
		   nullptr 4))))
	     (defun createModule ()
	       ,(set-members-clear`(,(g `module_compile_options)
				    :maxRegisterCount 50
				    :optLevel OPTIX_COMPILE_OPTIMIZATION_DEFAULT
				     :debugLevel OPTIX_COMPILE_DEBUG_LEVEL_NONE))
	       ,(set-members-clear`(,(g `pipeline_compile_options)
				     :traversableGraphFlags OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS
				     :usesMotionBlur false
				     :numPayloadValues 2
				     :numAttributeValues  2
				     :exceptionFlags     OPTIX_EXCEPTION_FLAG_NONE
				     :pipelineLaunchParamsVariableName (string "optixLaunchParams") 
      
				     ))
	       ,(set-members-clear`(,(g `pipeline_link_options)
				           
				     :overrideUsesMotionBlur  false
				     :maxTraceDepth           2))

	       
	       (let ((ptx ptx_code)
		     (log[2048])
		     (size_log (sizeof log)))
		 (declare (type char log[2048])
			  (type "const std::string" ptx))
		 ,(ox `(optixModuleCreateFromPTX
			,(g `oxctx)
			(ref ,(g `module_compile_options))
			(ref ,(g `pipeline_compile_options))
					;ptx_code
			(ptx.c_str)
			;(strlen ptx_code)
			(ptx.size)
			log
			&size_log
			(ref ,(g `module))))
		 (when (< 1 size_log)
		   ,(logprint "" `(size_log log)))))
	     ,@(loop for line in
		    `((raygen RayGen "__raygen__renderFrame")
		      (miss Miss "__miss__radiance")
		      (hitgroup HitGroup)) collect
		    (destructuring-bind  (e f &optional entry) line
		      (let ((var (g (format nil "~a_programs" e))))
		       `(defun ,(format nil "create~aPrograms" f) ()
			  (dot ,var (resize 1))
			  (let ((pg_options)
				(pg_desc))
			    (declare (type OptixProgramGroupOptions pg_options)
				     (type OptixProgramGroupDesc pg_desc))
			    ,(set-members-clear `(pg_options))
			    ,(set-members-clear `(pg_desc
						  :kind ,(string-upcase (format nil "OPTIX_PROGRAM_GROUP_KIND_~a" f))
						  
						  ,@(if (member e `(raygen miss))
							`(,(intern (string-upcase (format nil "~a.module" (string-downcase f)))) ,(g `module)
							   ,(intern (format nil "~a.entryFunctionName"
									   (string-downcase f)))
							   (string ,entry))
							`(,(intern (format nil "~a.moduleCH" (string-downcase f))) ,(g `module)
							   ,(intern (format nil "~a.moduleAH" (string-downcase f))) ,(g `module)
							   ,(intern (format nil "~a.entryFunctionNameCH"
									   (string-downcase f)))
							   (string "__closesthit__radiance")
							   ,(intern (format nil "~a.entryFunctionNameAH"
									    (string-downcase f)))
							   (string "__anyhit__radiance")))))
			    (let ((log[2048])
				  (size_log (sizeof log)))
			      (declare (type char log[2048])
				       )
			      ,(ox `(optixProgramGroupCreate
				     ,(g `oxctx)
				     (ref pg_desc)
				     1
				     (ref pg_options)
				     log
				     &size_log
				     (ref (aref ,var 0))))
			      (when (< 1 size_log)
				,(logprint "" `(size_log log)))))))))
	     (defun createPipeline ()
	       (let ((program_groups))
		 (declare (type "std::vector<OptixProgramGroup>" program_groups))
		 ,@(loop for e in `(raygen miss hitgroup) collect
			`(foreach (p ,(g (format nil "~a_programs" e)))
				  (program_groups.push_back p)))
		 (let ((log[2048])
		       (size_log (sizeof log)))
		   (declare (type char log[2048]))
			      ,(ox `(optixPipelineCreate
				     ,(g `oxctx)
				     (ref ,(g `pipeline_compile_options))
				     (ref ,(g `pipeline_link_options))
				     (program_groups.data)
				     (static_cast<int> (program_groups.size))
				     log
				     &size_log
				     (ref ,(g `pipeline))))
			      (when (< 1 size_log)
				,(logprint "" `(size_log log))))
		 ,(ox `(optixPipelineSetStackSize
			,(g `pipeline)
			(* 2 1024) ;; direct, invoked from ISor AH
 			(* 2 1024) ;; direct, invoked from RG MSor CH
			(* 2 1024) ;; continuation
			1 ;; maximum depth of traversable graph passed to trace
			))))
	     ,@(loop for e in `(raygen miss hitgroup) collect
		    `(defstruct0
			 ,(format nil "__align__(OPTIX_SBT_RECORD_ALIGNMENT) ~a_record_t" e)
			 ((aref header OPTIX_SBT_RECORD_HEADER_SIZE)
			  "__align__(OPTIX_SBT_RECORD_ALIGNMENT) char")
		       ,(case e
			  ('hitgroup `(object_id int))
			  (t `(data "void*")))
		 ))
	     (defun buildSBT ()
	       ,@(loop for e in `(raygen miss hitgroup) collect
		      (let ((records (format nil "~a_records" e))
			    (progs (g (format nil "~a_programs" e)))
			    (buffer (g (format nil "~a_records_buffer" e)))
			    (type (format nil "~a_record_t" e)))
			`(let ((,records))
			   (declare (type ,(format nil "std::vector<~a>" type)  ,records))
			 (dotimes (i (dot ,progs (size)))
			   (let ((rec))
			     (declare (type ,type rec))
			     ,(ox `(optixSbtRecordPackHeader (aref ,progs i)
							     (ref rec)))
			     (dot ,records (push_back rec))))
			 (dot ,buffer
			      (alloc_and_upload ,records))
			 ,(cond
			    ((eq e 'raygen)
			     `(setf (dot ,(g `shader_bindings_table)
					,(format nil "~aRecord" e))
				   (dot ,buffer
					(d_pointer))))
			    (t
			     `(do0
			       (setf (dot ,(g `shader_bindings_table)
					  ,(format nil "~aRecordBase" e))
				     (dot ,buffer
					  (d_pointer)))
			       (setf (dot ,(g `shader_bindings_table)
					  ,(format nil "~aRecordStrideInBytes" e))
				     (sizeof ,type))
			       (setf (dot ,(g `shader_bindings_table)
					  ,(format nil "~aRecordCount" e))
				     (static_cast<int> (dot ,records
							    (size)))))))))))
	     (defun render ()
	       (when (== 0 ,(g `launch_params.fbSize_x))
		 ,(logprint "can't launch because first resize hasn't happened" `())
		 (return))
	       (dot ,(g `launch_params_buffer)
		    (upload (ref ,(g `launch_params))
			    1))
	       (incf (dot ,(g `launch_params)
			  frameID))
	       ,(ox `(optixLaunch
		      ,(g `pipeline)
		      ,(g `stream)
		      (dot ,(g `launch_params_buffer)
			   (d_pointer))
		      (dot ,(g `launch_params_buffer)
			   _size_in_bytes)
		      (ref ,(g `shader_bindings_table))
		      ,(g `launch_params.fbSize_x)
		      ,(g `launch_params.fbSize_y)
		      1))
	       (progn
		 (cudaDeviceSynchronize)
		 ,(cu `(cudaGetLastError))))
	     (defun resize (x y)
	       (declare (type int x y))
	       (dot ,(g `color_buffer)
		    (resize (* x y (sizeof uint32_t))))
	       (setf ,(g `launch_params.fbSize_x) x
		     ,(g `launch_params.fbSize_y) y
		     ,(g `launch_params.colorBuffer)
		     (static_cast<uint32_t*> (dot ,(g `color_buffer)
						  _d_ptr)))
	       ;; reset camera in case aspect has changed
	       (set_camera ,(g `last_set_camera)))
	     (defun download_pixels (h_pixels)
	       (declare (type "uint32_t*" h_pixels))
	       (dot ,(g `color_buffer)
		    (download h_pixels
			      (* ,(g `launch_params.fbSize_x)
				 ,(g `launch_params.fbSize_y)
				 ))))
	     (defun set_camera (camera)
	       (declare (type "const camera_t&" camera))
	       
	       (setf ,(g `last_set_camera) camera)
	       (let ((cos_fov_y .66s0)
		     (aspect (/ (static_cast<float> ,(g `launch_params.fbSize_x))
				,(g `launch_params.fbSize_y))))
		 ,@(loop for (e f) in
			`((position camera.from)
			  (direction ("glm::normalize" (- camera.at camera.from)))
			  (horizontal (* cos_fov_y
					 aspect
					 ("glm::normalize"
					  ("glm::cross"
					   ,(g `launch_params.camera_direction)
					   camera.up))))
			  (vertical (* cos_fov_y
				       ("glm::normalize"
					("glm::cross"
					 ,(g `launch_params.camera_horizontal)
					 ,(g `launch_params.camera_direction))))))
		      collect
			`(setf (dot ,(g `launch_params) ,(format nil "camera_~a" e)) ,f))))

	     (defun buildAccel (model)
	       (declare (type "const triangle_mesh_t&" model)
			(values OptixTraversableHandle)
			)
	       ,(logprint "start building acceleration structure" `())
	       ,@(loop for e in `(vertex index) collect
		      `(dot ,(g (format nil "~a_buffer" e)) (alloc_and_upload
						       (dot model ,(format nil "_~a" e)))))
	       (let ((handle (curly 0))
		     (triangle_input (curly))
		     (d_vertices (dot ,(g `vertex_buffer) (d_pointer)))
		     (d_indices (dot ,(g `index_buffer) (d_pointer)))
		     (triangle_input_flags[] (curly 0)))
		 (declare (type OptixTraversableHandle handle)
			  (type OptixBuildInput triangle_input)
			  (type uint32_t triangle_input_flags[]))
		 ,(set-members `(triangle_input
				 :type OPTIX_BUILD_INPUT_TYPE_TRIANGLES
				 :triangleArray.vertexFormat OPTIX_VERTEX_FORMAT_FLOAT3
				 :triangleArray.vertexStrideInBytes (sizeof "glm::vec3")
				 :triangleArray.numVertices (static_cast<int> (model._vertex.size))
				 ;; FIXME reference or not (inconsistent in 04)
				 :triangleArray.vertexBuffers &d_vertices
				 :triangleArray.indexFormat OPTIX_INDICES_FORMAT_UNSIGNED_INT3
				 :triangleArray.indexStrideInBytes (sizeof "glm::ivec3")
				 :triangleArray.numIndexTriplets  (static_cast<int> (model._index.size))
				 :triangleArray.indexBuffer d_indices

				 :triangleArray.flags triangle_input_flags
				 :triangleArray.numSbtRecords 1
				 :triangleArray.sbtIndexOffsetBuffer 0
				 :triangleArray.sbtIndexOffsetSizeInBytes 0
				 :triangleArray.sbtIndexOffsetStrideInBytes 0
				 ))
		 (let ((accel_options (curly)))
		   (declare (type OptixAccelBuildOptions accel_options))
		   ,(set-members `(accel_options
				   :buildFlags (logior OPTIX_BUILD_FLAG_NONE
						       OPTIX_BUILD_FLAG_ALLOW_COMPACTION)
				   :motionOptions.numKeys 1
				   :operation OPTIX_BUILD_OPERATION_BUILD)))
		 (let ((blas_buffer_sizes))
		   (declare (type OptixAccelBufferSizes blas_buffer_sizes))
		   ,(ox `(optixAccelComputeMemoryUsage
			  ,(g `oxctx)
			  &accel_options
			  &triangle_input
			  1
			  &blas_buffer_sizes)))
		 ,(logprint "prepare compaction" `())
		 ;; prepare compaction
		 (let ((compacted_size_buffer)
		       (emit_desc))
		   (declare (type CUDABuffer compacted_size_buffer)
			    (type OptixAccelEmitDesc emit_desc))
		   (compacted_size_buffer.alloc (sizeof uint64_t))
		   ,(set-members `(emit_desc
				   :type OPTIX_PROPERTY_TYPE_COMPACTED_SIZE
				   :result (compacted_size_buffer.d_pointer))))
		 ,(logprint "execute build" `())
		 ;; execute build
		 (let ((temp_buffer)
		       (output_buffer))
		   (declare (type CUDABuffer temp_buffer output_buffer))
		   (temp_buffer.alloc blas_buffer_sizes.tempSizeInBytes)
		   (output_buffer.alloc blas_buffer_sizes.outputSizeInBytes)
		   ,(ox `(optixAccelBuild
			  ,(g `oxctx)
			  0 ;; stream
			  &accel_options
			  &triangle_input
			  1
			  (temp_buffer.d_pointer)
			  temp_buffer._size_in_bytes
			  (output_buffer.d_pointer)
			  output_buffer._size_in_bytes
			  &handle
			  &emit_desc
			  1)))
		 (progn
		 (cudaDeviceSynchronize)
		 ,(cu `(cudaGetLastError)))

		 ,(logprint "perform compaction" `())
		 ;; perform compaction
		 (let ((compacted_size))
		   (declare (type uint64_t compacted_size))
		   (compacted_size_buffer.download &compacted_size 1)
		   (dot ,(g `accel_buffer)
			(alloc compacted_size))
		   ,(ox `(optixAccelCompact
			  ,(g `oxctx)
			  0 ;; stream
			  handle
			  (,(g `accel_buffer.d_pointer))
			  ,(g `accel_buffer._size_in_bytes)
			  &handle
			  ))
		   (progn
		 (cudaDeviceSynchronize)
		 ,(cu `(cudaGetLastError))))

		 ;; clean up
		 ,(logprint "clean up" `())
		 (output_buffer.free)
		 (temp_buffer.free)
		 (compacted_size_buffer.free)
		 (return handle)
		 

		 ))

	     (defun initOptix (model)
	       (declare (type "const triangle_mesh_t&" model))
	       ,(logprint "initOptix" '())
	      #+nil (let ((choice 0))
		 (declare (type int choice))
		 ,(cu `(cudaChooseDevice &choice nullptr))
		 ,(logprint "cuda device" `(choice)))
	       ;,(cu `(cudaFree 0))
	       (let ((num_devices))
		 (declare (type int num_devices))
		 (cudaGetDeviceCount &num_devices))
	       (when (== 0 num_devices)
		 ,(logprint (string "FAIL: no cuda device")))
	       ,(ox `(optixInit))
	       (createContext)
	       (createModule)
	       (createRayGenPrograms)
	       (createMissPrograms)
	       (createHitGroupPrograms)
	       (setf (dot ,(g `launch_params)
			  traversable)
		     (buildAccel model))
	       (createPipeline)
	       (buildSBT)

	       (dot ,(g `launch_params_buffer)
		    (alloc (sizeof LaunchParams)))
	       )
	     (defun cleanupOptix ()
	       )
	     )))

  (define-module
      `(cuda_device_programs
	()
	(do0
	 (include <optix_device.h>)
	 " "

	 #+nil (include "LaunchParams.h")
	 #+nil
	 (defstruct0 LaunchParams
			(frameID int)
		      (colorBuffer uint32_t*)
		      (fbSize_x int)
		      (fbSize_y int)
		      ;(fbSize vec2i)
		      )

	 
	 (let ((optixLaunchParams))
	   (declare (type "extern \"C\" __constant__ LaunchParams" optixLaunchParams)))

	 "enum { SURFACE_RAY_TYPE=0, RAY_TYPE_COUNT };"

	 (defun unpack_pointer (i0 i1)
	   (declare (type uint32_t i0 i1)
		    (values "static __forceinline__ __device__ void*"))
	   (let ((uptr (logior (<< (static_cast<uint64_t> i0)
				   32)
			       i1)))
	     ;(declare (type "const uint64_t" uptr))
	     (return (reinterpret_cast<void*> uptr))))

	 (defun pack_pointer (ptr i0 i1)
	   (declare (type "uint32_t&" i0 i1)
		    (type void* ptr)
		    (values "static __forceinline__ __device__ void"))
	   
	   (let ((uptr (reinterpret_cast<uint64_t> ptr))
		 )
	     (setf i0 (>> uptr 32)
		   i1 (& uptr (hex #xffffffff)))))

	 (defun get_prd ()
	   (declare (values "template<typename T> static __forceinline__ __device__ T*"))
	   (let ((u0 (optixGetPayload_0))
		 (u1 (optixGetPayload_1)))
	     (return (reinterpret_cast<T*> (unpack_pointer u0 u1)))))

	 (defun random_color (i)
	   (declare (values "inline __device__ glm::vec3")
		    (type int i))
	   (let ((r (static_cast<int>
		     (+ (hex #x234235)
			(* 13 17 ("static_cast<unsigned>" i)))))
		 (g (static_cast<int>
		     (+ (hex #x773477)
			(* 7 3 5 ("static_cast<unsigned>" i)))))
		 (b (static_cast<int>
		     (+ (hex #x223766)
			(* 11 19 ("static_cast<unsigned>" i))))))
	     (return ("glm::vec3" (/ (logand r 255) 255s0)
			    (/ (logand g 255) 255s0)
			    (/ (logand b 255) 255s0)))))

	 
	 
	 (defun __closesthit__radiance ()
	   (declare (values "extern \"C\" __global__ void"))
	   (let ((id (optixGetPrimitiveIndex))
		 (prd (deref ("get_prd<glm::vec3>")))
		 )
	     (declare (type "glm::vec3&" prd))
	     (setf prd (random_color id))))
	 
	 (defun __anyhit__radiance ()
	   (declare (values "extern \"C\" __global__ void")))
	 (defun __miss__radiance ()
	   (declare (values "extern \"C\" __global__ void"))
	   (let (
		 (prd (deref ("get_prd<glm::vec3>")))
		 )
	     (declare (type "glm::vec3&" prd))
	     (setf prd ("glm::vec3" 1s0))))
	 (defun __raygen__renderFrame ()
	   (declare (values "extern \"C\" __global__ void"))
	   (let ((frameID optixLaunchParams.frameID)
		 (ix (dot (optixGetLaunchIndex) x))
		 (iy (dot (optixGetLaunchIndex) y))
		 ,@(loop for e in `(position direction horizontal vertical)
		      collect
			(let ((v (format nil "camera_~a" e)))
			  `(,v (dot optixLaunchParams ,v))))
		 (pixel_color_prd ("glm::vec3" 0s0)) ;; will be overwritten by hit or miss
		 (u0 (uint32_t 0))
		 (u1 (uint32_t 0))
		 (screen (/ ("glm::vec2"
			     (+ ix .5f)
			     (+ iy .5f))
			    ("glm::vec2"
			     (dot optixLaunchParams fbSize_x)
			     (dot optixLaunchParams fbSize_y))))
		 (ray_dir ("glm::normalize"
			   (+ camera_direction
			      (* camera_horizontal (- (aref screen 0) .5s0))
			      (* camera_vertical (- (aref screen 1) .5s0)))
			   ))
		 (fbIndex (+ ix
			     (* iy optixLaunchParams.fbSize_x))))
	     (declare (type "const int" frameID)))
	   (let ((pos (reinterpret_cast<float3*> &camera_position))
		 (dir (reinterpret_cast<float3*> &ray_dir)))
	    (optixTrace
	     optixLaunchParams.traversable
	     *pos
	     *dir
	     0s0 ;; tmin
	     1s20 ;; tmax
	     0s0  ;; ray time
	     (OptixVisibilityMask 255)
	     OPTIX_RAY_FLAG_DISABLE_ANYHIT
	     SURFACE_RAY_TYPE
	     RAY_TYPE_COUNT
	     SURFACE_RAY_TYPE
	     u0 u1))
	   (let (,@(loop for e in `(r g b) and i from 0 collect
			`(,e (static_cast<int> (* 255.99s0 (aref pixel_color_prd ,i)))))
		 (rgba (logior #xff000000 ;; fully opaque alpha
			       (<< r 0)
			       (<< g 8)
			       (<< b 16)))
		 ))
	   (setf (aref optixLaunchParams.colorBuffer fbIndex)
		 rgba)))))

  
  (progn
    (with-open-file (s (asdf:system-relative-pathname 'cl-cpp-generator2
						      (merge-pathnames #P"proto2.h"
								       *source-dir*))
		       :direction :output
		       :if-exists :supersede
		       :if-does-not-exist :create)
      (loop for e in (reverse *module*) and i from 0 do
	   (destructuring-bind (&key name code) e
	     (let ((cuda (cl-ppcre:scan "cuda" (string-downcase (format nil "~a" name)))))
	       (unless cuda
		 (emit-c :code code :hook-defun 
			 #'(lambda (str)
			     (format s "~a~%" str))))
	       
	       (write-source (asdf:system-relative-pathname
			      'cl-cpp-generator2
			      (format nil
				      "~a/vis_~2,'0d_~a.~a"
				      *source-dir* i name
				      (if cuda
					  "cu"
					  "cpp")))
			     code)))))
    (write-source (asdf:system-relative-pathname
		   'cl-cpp-generator2
		   (merge-pathnames #P"utils.h"
				    *source-dir*))
		  `(do0
		    "#ifndef UTILS_H"
		    " "
		    "#define UTILS_H"
		    " "
		    (include <vector>
			     <array>
			     <iostream>
			     <iomanip>)
		    
		    " "
		    (do0
		     
		     " "
		     ,@(loop for e in (reverse *utils-code*) collect
			  e)
			
		     
		     " "
		     
		     )
		    " "
		    "#endif"
		    " "))
    (write-source (asdf:system-relative-pathname 'cl-cpp-generator2 (merge-pathnames
								     #P"globals.h"
								     *source-dir*))
		  `(do0
		    "#ifndef GLOBALS_H"
		    " "
		    "#define GLOBALS_H"
		    " "
		    (include <GLFW/glfw3.h>)
		    
		    " "
		    " "
		    (include <thread>
			     <mutex>
			     <queue>
			     <deque>
			     <string>
			     <condition_variable>
			     <complex>)


		    (include <cuda_runtime.h>
			     <optix.h>
			     <optix_stubs.h>)

		    (include <glm/vec3.hpp>
			      <glm/mat3x3.hpp>)
		    
		    (include <cassert>)

		    (defstruct0 LaunchParams
			(frameID int)
		      (colorBuffer uint32_t*)
		      (fbSize_x int)
		      (fbSize_y int)
		      ,@(loop for e in `(position direction horizontal vertical) collect
			     `(,(format nil "camera_~a" e)  "glm::vec3"))
		      (traversable OptixTraversableHandle)
		      )

		    (defstruct0 camera_t
			,@(loop for e in `(from at up) collect
			       `(,e "glm::vec3")))

		    
		    (do0
		     (defclass linear_space_t ()
		       "public:"
		       (let ((vx)
			     (vy)
			     (vz))
			 (declare (type "glm::vec3" vx vy vz))))
		     (defclass affine_space_t ()
		       "public:"
		       (let ((l)
			     (p))
			 (declare (type "glm::vec3" p)
				  (type linear_space_t l)))))

		    (defun fma (a b c)
		      (declare (type "const glm::vec3&" a b c)
			       (values "inline glm::vec3"))
		      (return (+ (* a b) c)))
		    (defun xfm_point (m p)
			 (declare (type "const glm::vec3&" p)
				  (type "const affine_space_t&" m)
				  (values "inline glm::vec3"))
			 (let ((c (fma ("glm::vec3" (aref p 2))
						m.l.vz
						m.p))
			       (b (fma
				    ("glm::vec3" (aref p 1))
				    m.l.vy
				    c))
			       (a (fma
				   ("glm::vec3" (aref p 0))
				   m.l.vx
				   b)))
			  (return a)))
		    (do0
		     (defclass triangle_mesh_t ()
		       "public:"
		       (let ((_vertex)
			     (_index))
			 (declare (type "std::vector<glm::vec3>" _vertex)
				  (type "std::vector<glm::ivec3>" _index)))
		       (defun add_unit_cube (m)
			 (declare (type "const affine_space_t&" m))
			 (let ((first_vertex_id (static_cast<int>
						 (_vertex.size))))
			   ,@(loop for (x y z) in `((0 0 0)
						    (1 0 0)
						    (0 1 0)
						    (1 1 0)
						    (0 0 1)
						    (1 0 1)
						    (0 1 1)
						    (1 1 1))
				collect
				  `(_vertex.push_back
				    (xfm_point m ("glm::vec3" ,(* 1s0 x)
						      ,(* 1s0 y)
						      ,(* 1s0 z))))))
			 (let ((indices[] (curly 0 1 3  2 3 0
						 5 7 6  5 6 4
						 0 4 5  0 5 1
						 2 3 7  2 7 6
						 1 5 7  1 7 3
						 4 0 2  4 2 6)))
			   (declare (type int indices[]))
			   (dotimes (i 12)
			     (_index.push_back
			      (+ ("glm::ivec3"
				  (aref indices (+ 0 (* 3 i)))
				  (aref indices (+ 1 (* 3 i)))
				  (aref indices (+ 2 (* 3 i))))
				 first_vertex_id)))))
		       (defun add_cube (center size)
			 (declare  (type "const glm::vec3&"
					 center
					 size))
			 (let ((m
				(curly
				 (curly
				  ,@(loop for (e f) in '((x (1 0 0))
							 (y (0 1 0))
							 (z (0 0 1)))
				       collect
					 `("glm::vec3" ,@(loop for g in f collect
							      (if (eq g 0)
								  0
								  (format nil "size.~a" e))))))
				 (- center (* .5s0 size)))))
			   (declare (type "const affine_space_t" m))
			   #+nil
			   (do0
			    (setf m.p (- center (* .5s0 size))
				  )
			    ,@(loop for (e f) in '((x (1 0 0))
						   (y (0 1 0))
						   (z (0 0 1)))
				 collect
				   `(setf ,(format nil "m.l.v~a" e)
					  ("glm::vec3" ,@(loop for g in f collect
							      (if (eq g 0)
								  0
								  (format nil "size.~a" e)))))))
			   (add_unit_cube m)))))
		    
		    (do0
		     (defclass CUDABuffer ()
		       "public:"
		       (let ((_d_ptr)
			     (_size_in_bytes))
			 (declare (type void* _d_ptr)
				  (type size_t _size_in_bytes)))
		       (defun d_pointer ()
			 (declare (values CUdeviceptr))
			 (return (reinterpret_cast<CUdeviceptr> _d_ptr)))
		       (defun resize (size)
			 (declare (type size_t size))
			 (when _d_ptr
			   (free))
			 (alloc size))
		       (defun alloc (size)
			 (declare (type size_t size))
			 (assert (== nullptr _d_ptr))
			 (setf this->_size_in_bytes size)
			 ,(cu `(cudaMalloc (static_cast<void**> &_d_ptr)
					  _size_in_bytes)))
		       (defun free ()
			 ,(cu `(cudaFree _d_ptr))
			 (setf _d_ptr nullptr
			       _size_in_bytes 0))
		       (defun alloc_and_upload (vt)
			 (declare (type "const std::vector<T>&" vt)
				  (values "template<typename T> void"))
			 (alloc (* (vt.size)
				   (sizeof T)))
			 (upload ("static_cast<const T*>" (vt.data))
				 (vt.size)))
		       (defun upload (dat count)
			 (declare (type "const T*" dat)
				  (type size_t count)
				  (values "template<typename T> void"))
			 (assert (!= nullptr _d_ptr))
			 (assert (== _size_in_bytes (* count (sizeof T))))
			 ,(cu `(cudaMemcpy _d_ptr
					   ("static_cast<const void*>" dat)
					  (* count (sizeof T))
					  cudaMemcpyHostToDevice)))
		       (defun download (dat count)
			 (declare (type "T*" dat)
				  (type size_t count)
				  (values "template<typename T> void"))
			 (assert (!= nullptr _d_ptr))
			 (assert (== _size_in_bytes (* count (sizeof T))))
			 ,(cu `(cudaMemcpy  (static_cast<void*> dat)
					  _d_ptr
					  (* count (sizeof T))
					  cudaMemcpyDeviceToHost)))))
		    
		    (do0
		     "template <typename T, int MaxLen>"
		     (defclass FixedDequeTM "public std::deque<T>"
		       "// https://stackoverflow.com/questions/56334492/c-create-fixed-size-queue"
		       
		       "public:"
		       (let ((mutex))
			 (declare (type "std::mutex" mutex)))
		       (defun push_back (val)
			 (declare (type "const T&" val))
			 (when (== MaxLen (this->size))
			   (this->pop_front))
			 ("std::deque<T>::push_back" val))))
		    
		    " "
		    ,(emit-globals)
		    " "
		    "#endif"
		    " "))))

