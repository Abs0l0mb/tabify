'use strict';

import { 
    Block,
    Div,
    Api,
    Tools,
    Button
} from '@src/classes';

export class BasicTable extends Div {

    public table: Block;
    public currentRow: Block;
    public exportButton: Button;

    constructor(private parent: Block) {

        super('basic-table');

        this.table = new Block('table', {}, this);

        this.exportButton =  new Button({
            label: 'Export',
            class: 'download-button'
        }, this).onNative('click', this.downloadXLSX.bind(this));
    }

    /*
    **
    **
    */
    public attachOnDOM() {

        this.appendTo(this.parent);
    }

    /*
    **
    **
    */
    public addRow() : Block | null {

        this.currentRow = new Block('tr', {}, this.table);

        return this.currentRow;
    }

    /*
    **
    **
    */
    public addCell(value?: any, type: string = 'default', colspan: number = 0) : Block | null {

        if (!this.currentRow)
            return null;

        const cell = new Block('td', {
            class: type
        }, this.currentRow).html(value !== undefined ? value : '');

        if (colspan > 0) 
            cell.setAttribute('colspan', colspan);

        return cell;
    }

    /*
    **
    **
    */
    public async downloadXLSX() : Promise<void> {

        this.exportButton.load();

        let data: string[] = [];

        for (let tr of this.table.element.children) {

            const row: string[] = [];

            const excelCellLimit = 32767;
            const excelCellLimitMessage = '... Results ignored. Excel cannot exceed 32767 characters per cell.';

            for (let td of tr.children) {
                
                let text = td.innerText.replace('/n', '/r/n');

                text = td.innerText.replace(';', '|');

                if (text.length > excelCellLimit)
                    text = text.slice(0, excelCellLimit - excelCellLimitMessage.length) + excelCellLimitMessage;
                
                row.push(text);
            }

            data.push(row.join(';'));
        }

        let csvString = data.join('\r\n');

        let base64 = await Api.post('/csv-to-xlsx-base64', {
            csv: csvString
        });

        let blob = Tools.base64ToBlob(base64, 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet');

        let a = document.createElement("a");

        a.href = URL.createObjectURL(blob);

        a.download = "wft-export.xlsx";
        document.body.appendChild(a);
        a.click(); 
        document.body.removeChild(a);

        this.exportButton.unload();
    }

    /*
    **
    **
    */
    public ready() : void {

        setTimeout(() => {
            this.setData('ready', 1);
        }, 100);
    }
}