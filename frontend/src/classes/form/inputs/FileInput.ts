'use strict';

import {
    InputStructure,
    Block,
    Div
} from '@src/classes';

export interface FileInputData {
    label: string;
    value?: Date;
    mandatory?: boolean;
    class?: string;
}

export class FileInput extends InputStructure {

    private fileInput: Block;
    private inputLike: Div;
    private name: string | null = null;
    private base64: string | null = null;
    
    constructor(public data: FileInputData, parent?: Block) {

        super(data, parent);

        this.setCustomType('file');
        
        this.fileInput = new Block('input', {
            type: 'file'
        }, this.inputContainer);

        this.fileInput.onNative('change', this.onFile.bind(this));

        this.inputLike = new Div('input-like', this.inputContainer);
    }

    /*
    **
    **
    */
    public setText(text: string) : void {
        
        this.inputLike.write(text);
        this.setFilled(true);
    }

    /*
    **
    **
    */
    public setValue() : void {}

    /*
    **
    **
    */
    public getValue() : string | null {

        return this.getBase64();
    }

    /*
    **
    **
    */
    public getBase64() : string | null {

        return this.base64;
    }

    /*
    **
    **
    */
    public getName() : string | null {

        return this.name;
    }

    /*
    **
    **
    */
    private async onFile(event: any) : Promise<void> {

        const file = event.target.files[0];

        this.name = file.name;
        this.base64 = await new Promise((resolve, reject) => {

            const reader = new FileReader();

            reader.readAsDataURL(file);
            reader.onload = () => {
                resolve(typeof reader.result === 'string' ? reader.result.split('base64,')[1] : '');
            };
            reader.onerror = reject;
        });

        this.setText(file.name);
    }
}