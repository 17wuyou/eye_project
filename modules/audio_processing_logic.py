# modules/audio_processing_logic.py
import time, datetime, logging, os, asyncio
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor

from . import state_manager
from . import performance_utils
from . import client_actions
from . import android_commands
from . import event_recorder
from . import general_utils
from . import service_management 
from . import llm_service
from . import latency_logger
from audio_utils import is_audio_active_by_rms # Direct import for VAD

import configs

logger = logging.getLogger(__name__)

# This function will be submitted to the audio_executor
def _perform_audio_processing_async(
    client_uuid: str, 
    full_dialogue_audio_bytes: bytes,
    sample_rate: int,
    was_kws_triggered: bool,
    dialogue_start_client_ts: Optional[float],
    socketio_instance: Any, 
    llm_executor_instance: Optional[ThreadPoolExecutor],
    trace_id: str
):
    """
    Core asynchronous audio processing logic (Diarization, ASR, LLM trigger).
    This function is designed to run in a separate thread via audio_executor.
    """
    cycle_start_time_perf = time.perf_counter() # For measuring this async task's duration
    
    # Ensure services are available
    asr_service = service_management.asr_service_instance
    diar_service = service_management.diarization_service_instance

    if not asr_service:
        logger.warning(f"客户端 {client_uuid} [{trace_id}]: ASR 服务不可用，无法处理音频。")
        latency_logger.log_event(trace_id, client_uuid, 'audio_processing_task', 'ERROR', {'reason': 'ASR service unavailable'})
        if client_uuid in state_manager.llm_query_active: state_manager.llm_query_active[client_uuid] = False # Reset LLM flag
        return

    final_transcripts: List[str] = []
    diar_duration_ms = 0.0
    asr_total_duration_ms = 0.0

    if diar_service:
        latency_logger.log_event(trace_id, client_uuid, 'diarization', 'START')
        diar_start_perf = time.perf_counter()
        try:
            raw_speaker_segments = diar_service.process_audio(
                full_dialogue_audio_bytes, sample_rate
            )
        except Exception as e_diar:
            logger.error(f"客户端 {client_uuid} [{trace_id}]: Diarization处理时发生错误: {e_diar}", exc_info=True)
            raw_speaker_segments = []
        diar_duration_ms = (time.perf_counter() - diar_start_perf) * 1000
        latency_logger.log_event(trace_id, client_uuid, 'diarization', 'END', {'duration_ms': f'{diar_duration_ms:.2f}', 'segments_found': len(raw_speaker_segments)})

        if not raw_speaker_segments:
            logger.info(f"客户端 {client_uuid} [{trace_id}]: Diarization 未返回有效片段，将对整个音频进行 ASR。")
            # Fall through to ASR on full audio if diarization fails or returns no segments
        else:
            for i, (final_label, _, seg_bytes, seg_sr) in enumerate(raw_speaker_segments):
                seg_trace_id = f"{trace_id}-seg{i}"
                latency_logger.log_event(seg_trace_id, client_uuid, 'asr_transcription_segment', 'START', {'segment': i, 'speaker': final_label})
                asr_seg_start_perf = time.perf_counter()
                segment_text = ""
                try:
                    segment_text = asr_service.transcribe_bytes(seg_bytes, seg_sr)
                except Exception as e_asr_seg:
                    logger.error(f"客户端 {client_uuid} [{seg_trace_id}]: 片段ASR处理时发生错误: {e_asr_seg}", exc_info=True)
                
                asr_seg_duration_ms = (time.perf_counter() - asr_seg_start_perf) * 1000
                asr_total_duration_ms += asr_seg_duration_ms
                latency_logger.log_event(seg_trace_id, client_uuid, 'asr_transcription_segment', 'END', {'duration_ms': f'{asr_seg_duration_ms:.2f}', 'text_len': len(segment_text)})

                if not general_utils._is_text_no_speech(segment_text) and not general_utils._is_text_blocked(segment_text):
                    display_label = "你" if final_label == "user" else final_label
                    final_transcripts.append(f"{display_label}: {segment_text}")
    
    # If no diarization service OR diarization yielded no usable segments, run ASR on the whole audio
    if not diar_service or not final_transcripts: # Check final_transcripts too, as diar may run but find nothing to ASR
        if not diar_service:
             logger.info(f"客户端 {client_uuid} [{trace_id}]: Diarization 服务不可用，将对整个音频进行 ASR。")
        
        latency_logger.log_event(trace_id, client_uuid, 'asr_transcription_full', 'START')
        asr_full_start_perf = time.perf_counter()
        try:
            fallback_text = asr_service.transcribe_bytes(full_dialogue_audio_bytes, sample_rate)
        except Exception as e_asr_fallback:
            logger.error(f"客户端 {client_uuid} [{trace_id}]: 回退ASR处理时发生错误: {e_asr_fallback}", exc_info=True)
            fallback_text = ""
        asr_full_duration_ms = (time.perf_counter() - asr_full_start_perf) * 1000
        asr_total_duration_ms = asr_full_duration_ms # Overwrite if this path is taken
        latency_logger.log_event(trace_id, client_uuid, 'asr_transcription_full', 'END', {'duration_ms': f'{asr_full_duration_ms:.2f}', 'text_len': len(fallback_text)})
        
        if not general_utils._is_text_no_speech(fallback_text) and not general_utils._is_text_blocked(fallback_text):
            final_transcripts = [f"未知说话人: {fallback_text}"] # Replace, not append

    # --- Process and send results ---
    if final_transcripts:
        full_transcript_text_for_gui = "\n".join(final_transcripts)
        socketio_instance.emit('asr_update', {
            'sid': f"android_{client_uuid}",
            'text': full_transcript_text_for_gui,
            'is_final_segment': True
        }, room=state_manager.GUI_LISTENERS_ROOM)

        event_recorder.record_asr_for_event(client_uuid, final_transcripts)
        
        # (Save subtitles to file logic - kept as is)
        if configs.SAVE_SUBTITLES_TO_FILE and not configs.DATASET_SAVE_EVENTS:
            try:
                # ... (original subtitle saving code) ...
                timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                subtitle_filename = f"subtitles_{client_uuid}_{timestamp_str}.txt"
                subtitle_save_dir = getattr(configs, 'SUBTITLES_SAVE_DIR', "saved_subtitles")
                os.makedirs(subtitle_save_dir, exist_ok=True)
                subtitle_filepath = os.path.join(subtitle_save_dir, subtitle_filename)
                with open(subtitle_filepath, "w", encoding="utf-8") as f:
                    f.write(full_transcript_text_for_gui)
            except Exception as e_save_sub:
                logger.error(f"保存字幕文件失败: {e_save_sub}")


        # (Android DRAW_TEXT command generation - kept as is, but ensure it uses socketio_instance)
        TEXT_BASE_ID = f"{getattr(configs, 'ASR_TEXT_ID_PREFIX', 'asr_text_')}{client_uuid}_{int(time.time()*1000)}"
        parsed_dialogue_items = []
        # ... (original parsing and command generation logic) ...
        for ft_line in final_transcripts:
            parts = ft_line.split(":", 1)
            speaker = parts[0].strip()
            text_content = parts[1].strip() if len(parts) > 1 else ""
            if text_content:
                parsed_dialogue_items.append({'speaker_tag': speaker, 'text_content': text_content})

        consolidated_texts_map: Dict[str, List[str]] = {}
        for item in parsed_dialogue_items:
            tag = item['speaker_tag']
            if tag not in consolidated_texts_map: consolidated_texts_map[tag] = []
            consolidated_texts_map[tag].append(item['text_content'])
        
        lines_to_potentially_display = []
        # ... (Populate lines_to_potentially_display)
        if "你" in consolidated_texts_map:
            user_text_joined = '\n'.join(consolidated_texts_map['你'])
            lines_to_potentially_display.append({'full_text_to_draw': f"你: {user_text_joined}", 'color': [0, 200, 0, 255], 'is_user_line': True})
        other_speaker_tags_sorted = sorted([tag for tag in consolidated_texts_map if tag != "你"])
        for speaker_tag_other in other_speaker_tags_sorted:
            speaker_text_joined = '\n'.join(consolidated_texts_map[speaker_tag_other])
            lines_to_potentially_display.append({'full_text_to_draw': f"{speaker_tag_other}: {speaker_text_joined}",'color': [255, 165, 0, 255],'is_user_line': False})

        android_draw_commands = []
        current_draw_y = configs.ANDROID_SUBTITLE_BASE_Y_NORMALIZED
        num_lines_actually_drawn = 0
        # ... (Generate android_draw_commands based on lines_to_potentially_display and configs)
        user_line_data = next((line for line in lines_to_potentially_display if line['is_user_line']), None)
        if user_line_data and num_lines_actually_drawn < configs.ANDROID_SUBTITLE_MAX_LINES:
            cmd = {"type": "DRAW_TEXT","text_id": f"{TEXT_BASE_ID}_line{num_lines_actually_drawn}","text": user_line_data['full_text_to_draw'],"position_normalized": [0.5, current_draw_y],"color_rgba": user_line_data['color'],"size_sp": configs.ANDROID_SUBTITLE_TEXT_SIZE_SP,"duration_ms": configs.ANDROID_SUBTITLE_DURATION_MS}
            android_draw_commands.append(cmd)
            current_draw_y -= configs.ANDROID_SUBTITLE_LINE_HEIGHT_NORMALIZED
            num_lines_actually_drawn += 1
        for line_data in lines_to_potentially_display:
            if not line_data['is_user_line'] and num_lines_actually_drawn < configs.ANDROID_SUBTITLE_MAX_LINES:
                if current_draw_y < 0.05: break
                cmd = {"type": "DRAW_TEXT","text_id": f"{TEXT_BASE_ID}_line{num_lines_actually_drawn}","text": line_data['full_text_to_draw'],"position_normalized": [0.5, current_draw_y],"color_rgba": line_data['color'],"size_sp": configs.ANDROID_SUBTITLE_TEXT_SIZE_SP,"duration_ms": configs.ANDROID_SUBTITLE_DURATION_MS}
                android_draw_commands.append(cmd)
                current_draw_y -= configs.ANDROID_SUBTITLE_LINE_HEIGHT_NORMALIZED
                num_lines_actually_drawn += 1

        for cmd_to_send in android_draw_commands:
            if not android_commands._send_encrypted_command_to_android(client_uuid, cmd_to_send):
                logger.warning(f"发送ASR行 {cmd_to_send['text_id']} 到 android_{client_uuid} 失败")
    else:
         logger.info(f"客户端 {client_uuid} [{trace_id}]: 没有生成有效的最终转录字幕。")

    # --- LLM Trigger ---
    if was_kws_triggered and llm_executor_instance:
        logger.info(f"[{client_uuid}] [{trace_id}] KWS 已触发，准备启动 LLM 任务。")
        asr_text_for_llm = "\n".join(final_transcripts) if final_transcripts else ""
        last_frame_b64 = state_manager.last_video_frames.get(client_uuid)
        
        if asr_text_for_llm and last_frame_b64:
            if llm_executor_instance._shutdown:
                logger.warning(f"[{client_uuid}] [{trace_id}] LLM执行器已关闭，无法提交LLM任务。")
                if client_uuid in state_manager.llm_query_active: state_manager.llm_query_active[client_uuid] = False
            else:
                state_manager.llm_query_active[client_uuid] = True # Set flag before submitting
                llm_executor_instance.submit(
                    asyncio.run, # Still need asyncio.run if handle_llm_and_tts_task is async
                    llm_service.handle_llm_and_tts_task(
                        client_uuid=client_uuid,
                        asr_text=asr_text_for_llm,
                        image_b64=last_frame_b64,
                        socketio=socketio_instance, # Pass the correct SocketIO instance
                        trace_id=trace_id # Pass trace_id for continued logging
                    )
                )
        else:
            logger.warning(f"[{client_uuid}] [{trace_id}] KWS 已触发但无法启动 LLM 任务。缺少 ASR 结果 ({bool(asr_text_for_llm)}) 或视频帧 ({bool(last_frame_b64)})。")
            if client_uuid in state_manager.llm_query_active: state_manager.llm_query_active[client_uuid] = False
    elif was_kws_triggered and not llm_executor_instance:
        logger.warning(f"[{client_uuid}] [{trace_id}] KWS 已触发但LLM执行器未配置，跳过LLM任务。")
        if client_uuid in state_manager.llm_query_active: state_manager.llm_query_active[client_uuid] = False


    # --- Performance Logging ---
    task_duration_ms = (time.perf_counter() - cycle_start_time_perf) * 1000
    end_to_end_latency_ms = -1
    if dialogue_start_client_ts:
        # Assuming dialogue_start_client_ts is a Unix timestamp (seconds)
        end_to_end_latency_ms = (time.time() - dialogue_start_client_ts) * 1000
            
    latency_logger.log_event(
        trace_id, client_uuid, 'audio_processing_task', 'END', 
        {
            'task_duration_ms': f'{task_duration_ms:.2f}', # Duration of this async function
            'diar_duration_ms': f'{diar_duration_ms:.2f}',
            'asr_total_duration_ms': f'{asr_total_duration_ms:.2f}',
            'end_to_end_latency_ms': f'{end_to_end_latency_ms:.2f}' if end_to_end_latency_ms != -1 else 'N/A'
        }
    )
    # logger.info(f"客户端 {client_uuid} [{trace_id}]: 异步音频处理完成，任务耗时 {task_duration_ms:.2f}ms。")

    # Update detailed performance cycle data if enabled
    if configs.ENABLE_PERFORMANCE_MONITORING:
        perf_lock = state_manager.client_data_locks.get(client_uuid, {}).get("performance_lock")
        if perf_lock:
            with perf_lock:
                if client_uuid in state_manager.client_performance_data:
                    p_data = state_manager.client_performance_data[client_uuid]
                    p_data["last_diarization_duration_ms"] = diar_duration_ms
                    p_data["last_asr_total_duration_ms"] = asr_total_duration_ms
                    # last_total_processing_wall_time_ms is tricky here as this is an async task.
                    # This measures the task itself, not necessarily the "wall time" from the perspective of the calling code.
                    # We can log it as the task's own processing time.
                    p_data["last_task_internal_processing_time_ms"] = task_duration_ms 
                    if dialogue_start_client_ts and p_data.get("cycle_server_recv_wall_time") is not None:
                        p_data["latency_client_to_server_processed_ms"] = (time.time() - dialogue_start_client_ts) * 1000
        else:
            logger.warning(f"[{client_uuid}] 异步音频处理: 无法获取性能锁来更新耗时。")


def process_accumulated_audio_wrapper(
    client_uuid: str, 
    socketio: Any, 
    llm_executor: Optional[ThreadPoolExecutor],
    audio_executor: Optional[ThreadPoolExecutor], # Added audio_executor
    force_process: bool = False
):
    """
    Wrapper function that decides if audio processing is needed and submits it to audio_executor.
    This function itself should be quick and non-blocking.
    """
    if client_uuid not in state_manager.client_audio_accumulators:
        logger.warning(f"客户端 {client_uuid}: 音频累积器未找到，无法处理累积音频。")
        return

    acc = state_manager.client_audio_accumulators[client_uuid]
    
    should_process = False
    reason = ""
    current_time_ns = time.monotonic_ns()
    # Ensure dialogue_last_speech_time_ns exists, else default to current_time_ns
    last_speech_time_ns = acc.get('dialogue_last_speech_time_ns', current_time_ns)
    dialogue_silence_duration_ms = (current_time_ns - last_speech_time_ns) / 1_000_000

    if force_process:
        if acc['dialogue_buffer']:
            should_process = True
            reason = "强制处理_清理"
    elif not acc['dialogue_buffer']:
        # If buffer is empty and it's been silent for a while, reset last speech time to prevent immediate trigger on next sound
        if dialogue_silence_duration_ms > configs.DIALOGUE_SILENCE_TIMEOUT_MS * 2: # Arbitrary factor
             acc['dialogue_last_speech_time_ns'] = current_time_ns
        return # Nothing to process
    elif dialogue_silence_duration_ms > configs.DIALOGUE_SILENCE_TIMEOUT_MS and not acc['vad_is_active']:
        should_process = True
        reason = f"对话静音超时 ({dialogue_silence_duration_ms:.0f}ms)"
    elif acc['dialogue_accumulated_chunks_count'] >= configs.MAX_ACCUMULATED_CHUNKS_FOR_DIALOGUE:
        should_process = True
        reason = f"达到最大对话累积块数 ({acc['dialogue_accumulated_chunks_count']} 块)"

    if should_process and acc['dialogue_buffer']:
        trace_id = latency_logger.generate_trace_id()
        latency_logger.log_event(trace_id, client_uuid, 'audio_processing_decision', 'SUBMIT', {'reason': reason, 'queue_len_dialogue_buffer': len(acc['dialogue_buffer'])})
        
        full_dialogue_audio_bytes = b''.join(list(acc['dialogue_buffer']))
        was_kws_triggered = acc.get('kws_was_triggered', False)
        dialogue_start_client_ts = acc.get('dialogue_start_client_timestamp')

        client_actions.initialize_client_accumulator(client_uuid) # Reset accumulator immediately

        if not full_dialogue_audio_bytes:
            logger.warning(f"客户端 {client_uuid} [{trace_id}]: 累积音频为空，不处理。")
            latency_logger.log_event(trace_id, client_uuid, 'audio_processing_task', 'SKIP', {'reason': 'empty audio buffer after dequeue'})
            return

        general_utils._save_app_level_audio(client_uuid, full_dialogue_audio_bytes, "dialogue_for_processing")
        
        if audio_executor and not audio_executor._shutdown:
            audio_executor.submit(
                _perform_audio_processing_async,
                client_uuid,
                full_dialogue_audio_bytes,
                configs.AUDIO_SAMPLE_RATE,
                was_kws_triggered,
                dialogue_start_client_ts,
                socketio,
                llm_executor,
                trace_id
            )
        else: # Fallback to synchronous execution if executor is not available or shut down
            logger.warning(f"客户端 {client_uuid} [{trace_id}]: 音频处理执行器不可用或已关闭。同步执行音频处理。")
            _perform_audio_processing_async(
                client_uuid, full_dialogue_audio_bytes, configs.AUDIO_SAMPLE_RATE,
                was_kws_triggered, dialogue_start_client_ts,
                socketio, llm_executor, trace_id
            )
            
    elif not acc['dialogue_buffer'] and not acc['vad_is_active']:
        # If buffer became empty (e.g., due to race condition with reset) and VAD is not active,
        # update last speech time if it's been silent for a while.
        if dialogue_silence_duration_ms > configs.DIALOGUE_SILENCE_TIMEOUT_MS * 1.5:
            acc['dialogue_last_speech_time_ns'] = current_time_ns