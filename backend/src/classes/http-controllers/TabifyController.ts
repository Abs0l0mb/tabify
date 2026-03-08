'use strict';

import {
    BackendHttpServer,
    HttpRequest,
    HttpResponse,
    PublicError
} from '@src/classes';

const PYTHON_SERVICE_URL = process.env.TABIFY_SERVICE_URL ?? 'http://127.0.0.1:5000';

export class TabifyController {

    constructor(server: BackendHttpServer) {

        server.post('/tabify', this.tabify);
    }

    /*
    **
    **
    */
    private async tabify(request: HttpRequest, response: HttpResponse): Promise<void> {

        const params = request.getParameters();

        const midi_base64: string | undefined = params.midi_base64;
        const midi_name:   string             = params.midi_name ?? 'input.mid';
        const step:        number             = params.step  ?? 60;
        const gpq:         number             = params.gpq   ?? 960;
        const tempo:       number             = params.tempo ?? 120;

        if (!midi_base64)
            throw new PublicError('missing-midi-file');

        let midiBuffer: Buffer;

        try {
            midiBuffer = Buffer.from(midi_base64, 'base64');
        } catch {
            throw new PublicError('invalid-midi-file');
        }

        // ── Call Python service ───────────────────────────────────────

        const formData = new FormData();
        formData.append('midi', new Blob([midiBuffer], { type: 'audio/midi' }), midi_name);
        formData.append('step',  String(step));
        formData.append('gpq',   String(gpq));
        formData.append('tempo', String(tempo));

        let pyResponse: Response;

        try {
            pyResponse = await fetch(`${PYTHON_SERVICE_URL}/tabify`, {
                method: 'POST',
                body:   formData
            });
        } catch {
            throw new PublicError('python-service-unavailable');
        }

        if (!pyResponse.ok) {
            let errorMsg = 'python-service-error';
            try {
                const data: any = await pyResponse.json();
                if (typeof data.error === 'string')
                    errorMsg = data.error;
            } catch {}
            throw new PublicError(errorMsg);
        }

        // ── Stream GP5 back to frontend ───────────────────────────────

        const gp5Buffer = Buffer.from(await pyResponse.arrayBuffer());
        const gp5Name   = midi_name.replace(/\.midi?$/i, '') + '.gp5';

        response.setHeader('Content-Disposition', `attachment; filename="${gp5Name}"`);
        response.sendContent('application/octet-stream', gp5Buffer);
    }
}
